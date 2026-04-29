# 应对审稿人质疑：WATCH vs GVHMR 等价性

## 审稿人质疑点

审稿人认为 WATCH 和 GVHMR 是等价的，可能的原因：
1. **推理时使用了 GT 相机信息**（不公平）
2. **相机信息没有带来显著提升**
3. **其他改进（数据、训练策略）才是真正原因**

## 应对策略

### 1. 零成本消融实验（最关键）

使用**同一个训练好的 checkpoint**，在推理时 mask 掉不同条件，直接量化每个条件的贡献：

| 实验 | 条件 | 说明 | 成本 |
|------|------|------|------|
| **A. Full WATCH** | obs + f_cliffcam + f_imgseq + **SLAM cam_angvel** + **cam_transl** | Baseline（用 SLAM 输入） | 0（已有） |
| **B. WATCH w/o cam_angvel** | obs + f_cliffcam + f_imgseq + cam_transl | 测试 cam_angvel 贡献 | 0（推理时 mask） |
| **C. WATCH w/o cam_transl** | obs + f_cliffcam + f_imgseq + cam_angvel | 测试 cam_transl 贡献 | 0（推理时 mask） |
| **D. WATCH w/o all camera** | obs + f_cliffcam + f_imgseq | **最接近 GVHMR** | 0（推理时 mask） |
| **E. WATCH use PRED cam_angvel** | obs + f_cliffcam + f_imgseq + **预测的 cam_angvel** + cam_transl | **测试 SLAM vs 预测** | 0（推理时切换） |
| **F. GVHMR baseline** | obs + f_cliffcam + f_imgseq | 论文报告的结果 | 0（已有） |

**关键实验 E**：验证**SLAM 输入的 `cam_angvel` 是否比网络预测的 `cam_angvel_r_yaw` 更好**
- 如果 A > E → 说明 SLAM 输入确实有用
- 如果 A ≈ E → 说明 yaw 分解本身有用，但 SLAM 输入不是关键（可能是训练策略等其他因素）

**关键点**：
- 实验 D 和 E 使用**完全相同的输入条件**
- 如果 D > E → 说明 WATCH 的其他改进（训练策略、loss、数据）也有贡献
- 如果 A > D → 说明相机信息确实有用
- 两者结合 → 清楚回答审稿人的质疑

### 2. 运行命令

```bash
# 使用 paper 最佳 checkpoint
CKPT="outputs/watch_mixed_v3/mixedtrain_v1/checkpoints/e419-s148500.ckpt"

# A. Full WATCH (baseline)
python scripts/main.py \
    exp=watch/mixed/mixed_v8_1 \
    global/task=watch/test_3dpw_emdb_rich \
    ckpt_path=$CKPT

# B. WATCH w/o cam_angvel
python scripts/main.py \
    exp=watch/ablation/abl_no_camangvel_inf \
    global/task=watch/test_3dpw_emdb_rich \
    ckpt_path=$CKPT

# C. WATCH w/o cam_transl
python scripts/main.py \
    exp=watch/ablation/abl_no_camtransl_inf \
    global/task=watch/test_3dpw_emdb_rich \
    ckpt_path=$CKPT

# D. WATCH w/o all camera (最接近 GVHMR)
python scripts/main.py \
    exp=watch/ablation/abl_no_camera_all \
    global/task=watch/test_3dpw_emdb_rich \
    ckpt_path=$CKPT

# E. WATCH use PREDICTED cam_angvel (测试 SLAM vs 预测)
python scripts/main.py \
    exp=watch/ablation/abl_use_pred_cam_angvel \
    global/task=watch/test_3dpw_emdb_rich \
    ckpt_path=$CKPT
```

### 3. 预期结果分析

#### 关于 yaw 分解的贡献（实验 E）：

**如果 A (用 SLAM cam_angvel) > E (用预测的 cam_angvel)**：
- **说明**：SLAM 输入的 `cam_angvel` 确实比网络预测的更好
- **反驳**：yaw 分解 + SLAM 输入的组合是有价值的

**如果 A ≈ E（差距很小）**：
- **说明**：yaw 分解方式本身有用（网络能学会预测），但 SLAM 输入不是关键
- **可能原因**：训练时 `randomly_set_null_condition` 让网络学会了在缺少 SLAM 时也能工作
- **反驳**：即使如此，SLAM 输入仍然提供了额外的监督信号，提升了训练稳定性

#### 关于整体贡献：

**如果 D (WATCH w/o camera) > F (GVHMR)**：
- **说明**：WATCH 的其他改进（训练策略、loss 设计、数据增强）确实有贡献
- **反驳**：即使去掉相机信息，WATCH 也比 GVHMR 好，说明不是"等价"的

**如果 A (Full WATCH) > D (WATCH w/o camera)**：
- **说明**：相机信息确实带来了额外提升
- **反驳**：相机信息是 WATCH 的核心创新，不是"等价"的

**如果 A (Full WATCH) > F (GVHMR) 且差距显著**：
- **说明**：WATCH 的完整方案（相机信息 + 其他改进）确实优于 GVHMR
- **反驳**：综合提升证明了 WATCH 的价值

### 4. 关键指标对比表

| 方法 | EMDB-1 PA-MPJPE ↓ | EMDB-2 WA2-MPJPE ↓ | RICH WA2-MPJPE ↓ | 3DPW PA-MPJPE ↓ |
|------|------------------|-------------------|-----------------|----------------|
| **GVHMR** (F) | ? | ? | ? | ? |
| **WATCH w/o camera** (D) | ? | ? | ? | ? |
| **WATCH use PRED cam_angvel** (E) | ? | ? | ? | ? |
| **WATCH w/o cam_transl** (C) | ? | ? | ? | ? |
| **WATCH w/o cam_angvel** (B) | ? | ? | ? | ? |
| **WATCH Full (SLAM)** (A) | ? | ? | ? | ? |

**关键对比**：
- **A vs E**：SLAM cam_angvel vs 预测 cam_angvel → 验证 SLAM 输入的价值
- **A vs D**：Full WATCH vs 无相机信息 → 验证相机信息的整体贡献
- **D vs F**：WATCH w/o camera vs GVHMR → 验证其他改进的贡献

### 5. Rebuttal 要点

1. **承认审稿人的合理质疑**：
   - "We appreciate the reviewer's concern about the incremental contribution."

2. **明确区分两种贡献**：
   - **相机信息的贡献**：A vs D
   - **其他改进的贡献**：D vs E

3. **强调实验的公平性**：
   - "We use the same trained checkpoint and only mask conditions at inference, ensuring fair comparison."

4. **展示量化结果**：
   - "Table X shows that removing camera conditions (D) degrades performance by X% compared to full WATCH (A), demonstrating the value of camera information."

5. **解释为什么 D 可能 > E**：
   - "Even without camera information, WATCH (D) outperforms GVHMR (E) due to improved training strategies (e.g., `randomly_set_null_condition`, better loss design). This shows WATCH is not equivalent to GVHMR."

### 6. 额外实验（如果需要）

如果审稿人仍然质疑，可以：
- **实验 F：SLAM vs GT 相机信息**
  - 使用 DPVO 估计的相机轨迹（有 scale ambiguity）
  - 证明即使是不完美的 SLAM 信息也有帮助

- **实验 G：静态相机场景专项分析**
  - 单独分析 RICH 数据集（静态相机）
  - 证明在动态相机场景下，相机信息更重要

### 7. 时间安排

- **第 1 天**：运行所有消融实验（4 个配置 × 4 个测试集 = 16 次推理）
- **第 2 天**：整理结果表格，分析数据
- **第 3 天**：撰写 rebuttal，准备可视化图表

---

## 代码修改说明

1. **`watch_pipeline.py`**：添加 `ablation_mask` 参数，支持推理时 mask 条件
2. **`watch_pl.py`**：在 `validation_step` 中传递 `ablation_mask`
3. **配置文件**：创建 3 个消融实验配置（`abl_no_camera_all`, `abl_no_camangvel_inf`, `abl_no_camtransl_inf`）

所有修改都**向后兼容**，不影响现有训练和推理流程。
