import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from time import time
import json
from pathlib import Path

from core.configs import MainStore, builds


class TrainingProfiler(pl.Callback):
    def __init__(self, output_dir="profiling_results"):
        """
        训练流程性能分析器
        记录整个训练流程各环节的耗时
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 计时器状态
        self.timers = {}
        self.profiling_data = {
            "data_preparation": None,
            "model_preparation": None,
            "trainer_setup": None,
            "training_phase": None,
            "testing_phase": None,
            "total_time": None
        }
        
        self.epoch_times = []
        self.batch_times = []
        self.batch_stage_times = []  # 存储每个batch的各阶段耗时

    def start_timer(self, name):
        """开始计时"""
        self.timers[name] = time()

    def stop_timer(self, name):
        """停止计时并记录耗时"""
        if name in self.timers:
            elapsed = time() - self.timers[name]
            self.profiling_data[name] = elapsed
            return elapsed
        return 0

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        """训练开始时的回调"""
        self.start_timer("total_time")
        self.start_timer("data_preparation")

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """训练阶段开始时的回调"""
        # 数据准备完成
        data_prep_time = self.stop_timer("data_preparation")
        
        # 开始模型准备计时
        self.start_timer("model_preparation")

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        """每个训练epoch开始时的回调"""
        if trainer.current_epoch == 0:
            # 第一个epoch开始时，模型准备完成
            model_prep_time = self.stop_timer("model_preparation")
            
            # 开始训练器配置计时
            self.start_timer("trainer_setup")
        
        # 开始epoch计时
        self.start_timer(f"epoch_{trainer.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        """每个训练epoch结束时的回调"""
        # 记录epoch耗时
        epoch_time = self.stop_timer(f"epoch_{trainer.current_epoch}")
        self.epoch_times.append(epoch_time)
        
        if trainer.current_epoch == 0:
            # 第一个epoch结束时，训练器配置完成
            trainer_setup_time = self.stop_timer("trainer_setup")
            
            # 开始训练阶段计时
            self.start_timer("training_phase")

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """每个训练batch开始时的回调"""
        self.start_timer(f"batch_{batch_idx}")
        # 开始batch内部阶段计时
        self.start_timer(f"batch_{batch_idx}_forward")
        self.start_timer(f"batch_{batch_idx}_data_to_device")

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """每个训练batch结束时的回调"""
        # 记录batch内部各阶段耗时
        data_to_device_time = self.stop_timer(f"batch_{batch_idx}_data_to_device")
        forward_time = self.stop_timer(f"batch_{batch_idx}_forward")
        
        # 开始反向传播计时
        self.start_timer(f"batch_{batch_idx}_backward")
        
        # 记录batch总时间
        batch_time = self.stop_timer(f"batch_{batch_idx}")
        self.batch_times.append(batch_time)
        
        # 记录各阶段耗时
        stage_times = {
            "batch_idx": batch_idx,
            "epoch": trainer.current_epoch,
            "total_time": batch_time,
            "data_to_device": data_to_device_time,
            "forward": forward_time,
            "backward": 0.0,  # 将在on_after_backward中更新
            "optimizer_step": 0.0  # 将在on_before_optimizer_step中更新
        }
        self.batch_stage_times.append(stage_times)
        
        # 实时打印batch耗时信息
        self.print_batch_profiling(trainer, batch_idx, stage_times)

    @rank_zero_only
    def on_test_start(self, trainer, pl_module):
        """测试阶段开始时的回调"""
        # 训练阶段结束
        training_time = self.stop_timer("training_phase")
        
        # 开始测试阶段计时
        self.start_timer("testing_phase")

    @rank_zero_only
    def on_after_backward(self, trainer, pl_module):
        """反向传播完成后的回调"""
        # 使用更可靠的方式获取当前batch索引
        # 从batch_stage_times中获取最新的batch索引
        if self.batch_stage_times:
            current_batch = self.batch_stage_times[-1]
            batch_idx = current_batch["batch_idx"]
            
            # 记录反向传播耗时
            backward_time = self.stop_timer(f"batch_{batch_idx}_backward")
            
            # 更新当前batch的反向传播时间
            current_batch["backward"] = backward_time
            
            # 开始优化器步骤计时
            self.start_timer(f"batch_{batch_idx}_optimizer")

    @rank_zero_only
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """优化器步骤开始前的回调"""
        # 使用更可靠的方式获取当前batch索引
        # 从batch_stage_times中获取最新的batch索引
        if self.batch_stage_times:
            current_batch = self.batch_stage_times[-1]
            batch_idx = current_batch["batch_idx"]
            
            # 记录优化器步骤耗时
            optimizer_time = self.stop_timer(f"batch_{batch_idx}_optimizer")
            
            # 更新当前batch的优化器时间
            current_batch["optimizer_step"] = optimizer_time

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        """测试阶段结束时的回调"""
        # 测试阶段结束
        testing_time = self.stop_timer("testing_phase")
        
        # 总时间
        total_time = self.stop_timer("total_time")
        
        # 保存分析结果
        self.save_profiling_results(trainer)

    def save_profiling_results(self, trainer):
        """保存性能分析结果"""
        # 计算batch阶段统计信息
        stage_stats = {}
        if self.batch_stage_times:
            stages = ["data_to_device", "forward", "backward", "optimizer_step"]
            for stage in stages:
                stage_times = [batch[stage] for batch in self.batch_stage_times if stage in batch]
                if stage_times:
                    stage_stats[f"avg_{stage}_time"] = sum(stage_times) / len(stage_times)
                    stage_stats[f"min_{stage}_time"] = min(stage_times)
                    stage_stats[f"max_{stage}_time"] = max(stage_times)
        
        results = {
            "profiling_summary": self.profiling_data,
            "epoch_times": self.epoch_times,
            "batch_times": self.batch_times,
            "batch_stage_times": self.batch_stage_times,
            "statistics": {
                "total_epochs": len(self.epoch_times),
                "total_batches": len(self.batch_times),
                "avg_epoch_time": sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0,
                "avg_batch_time": sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0,
                "min_batch_time": min(self.batch_times) if self.batch_times else 0,
                "max_batch_time": max(self.batch_times) if self.batch_times else 0,
                **stage_stats
            }
        }
        
        # 保存到文件
        output_file = self.output_dir / f"training_profiling_{int(time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 打印摘要信息
        self.print_profiling_summary(results)

    def print_batch_profiling(self, trainer, batch_idx, stage_times):
        """实时打印batch耗时信息"""
        # 计算当前epoch的统计信息
        current_epoch = trainer.current_epoch
        epoch_batch_times = []
        
        # 获取当前epoch的所有batch时间
        for i, time_val in enumerate(self.batch_times):
            if i >= len(self.batch_times) - (batch_idx + 1):
                epoch_batch_times.append(time_val)
        
        # 计算统计信息
        if epoch_batch_times:
            avg_batch_time = sum(epoch_batch_times) / len(epoch_batch_times)
            min_batch_time = min(epoch_batch_times)
            max_batch_time = max(epoch_batch_times)
        else:
            avg_batch_time = stage_times["total_time"]
            min_batch_time = stage_times["total_time"]
            max_batch_time = stage_times["total_time"]
        
        # 格式化打印信息 - 显示各阶段耗时
        print(f"\r[Epoch {current_epoch:3d} | Batch {batch_idx:4d}] "
              f"总: {stage_times['total_time']:.3f}s | "
              f"数据: {stage_times['data_to_device']:.3f}s | "
              f"前向: {stage_times['forward']:.3f}s | "
              f"反向: {stage_times['backward']:.3f}s | "
              f"优化: {stage_times['optimizer_step']:.3f}s | "
              f"平均: {avg_batch_time:.3f}s", 
              end="", flush=True)

    def print_profiling_summary(self, results):
        """打印性能分析摘要"""
        summary = results["profiling_summary"]
        stats = results["statistics"]
        
        print("\n" + "="*60)
        print("训练流程性能分析报告")
        print("="*60)
        
        if summary["data_preparation"]:
            print(f"数据准备耗时: {summary['data_preparation']:.2f}秒")
        if summary["model_preparation"]:
            print(f"模型准备耗时: {summary['model_preparation']:.2f}秒")
        if summary["trainer_setup"]:
            print(f"训练器配置耗时: {summary['trainer_setup']:.2f}秒")
        if summary["training_phase"]:
            print(f"训练阶段耗时: {summary['training_phase']:.2f}秒")
        if summary["testing_phase"]:
            print(f"测试阶段耗时: {summary['testing_phase']:.2f}秒")
        if summary["total_time"]:
            print(f"总耗时: {summary['total_time']:.2f}秒")
        
        print(f"\n训练统计:")
        print(f"  总轮次: {stats['total_epochs']}")
        print(f"  总批次: {stats['total_batches']}")
        print(f"  平均每轮耗时: {stats['avg_epoch_time']:.2f}秒")
        print(f"  平均每批次耗时: {stats['avg_batch_time']:.2f}秒")
        print(f"  最快批次: {stats['min_batch_time']:.4f}秒")
        print(f"  最慢批次: {stats['max_batch_time']:.4f}秒")
        
        # 显示batch阶段统计信息
        if 'avg_data_to_device_time' in stats:
            print(f"\nBatch阶段耗时统计:")
            print(f"  数据加载: {stats['avg_data_to_device_time']:.3f}s (min: {stats['min_data_to_device_time']:.3f}s, max: {stats['max_data_to_device_time']:.3f}s)")
            print(f"  前向传播: {stats['avg_forward_time']:.3f}s (min: {stats['min_forward_time']:.3f}s, max: {stats['max_forward_time']:.3f}s)")
            print(f"  反向传播: {stats['avg_backward_time']:.3f}s (min: {stats['min_backward_time']:.3f}s, max: {stats['max_backward_time']:.3f}s)")
            print(f"  优化器更新: {stats['avg_optimizer_step_time']:.3f}s (min: {stats['min_optimizer_step_time']:.3f}s, max: {stats['max_optimizer_step_time']:.3f}s)")
        
        print("="*60)


group_name = "callbacks/training_profiler"
base = builds(TrainingProfiler, populate_full_signature=True)
MainStore.store(name="base", node=base, group=group_name)