#!/bin/bash
# ============================================================
# WATCH Ablation & Improvement Experiments Launcher
# ============================================================
# Usage:
#   bash run_ablation.sh           # run all 8 experiments
#   bash run_ablation.sh abl       # run only 6 ablation experiments
#   bash run_ablation.sh imp       # run only 2 improvement experiments
#   bash run_ablation.sh single 0 abl_no_camtraj  # run single experiment on GPU 0
# ============================================================

set -e
cd "$(dirname "$0")"

CONDA_ENV="watch"
PYTHON="/home/nfs/hzy/envs/${CONDA_ENV}/bin/python"
SCRIPT="scripts/main.py"

# ===== Experiment definitions =====
# Format: "GPU_ID  EXP_CONFIG_NAME  DESCRIPTION"
ABL_EXPERIMENTS=(
    "0  abl_no_camtraj    A1:w/o_cam_traj"
    "1  abl_no_imgseq     A2:w/o_imgseq"
    "2  abl_no_camangvel   A3:w/o_cam_angvel"
    "3  abl_mutual_infer   A4:w/_mutual_infer"
    "4  abl_no_staticconf  A5:w/o_static_conf"
    "5  abl_no_globaltraj  A6:w/o_global_traj"
)

IMP_EXPERIMENTS=(
    "6  imp_gate_fusion    IMP:gate_fusion"
    "7  imp_adaln_fusion   IMP:adaln_fusion"
)

run_experiment() {
    local gpu_id=$1
    local exp_name=$2
    local desc=$3
    local log_file="outputs/watch_ablation/${exp_name}/train_watch.log"

    mkdir -p "outputs/watch_ablation/${exp_name}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${desc} on GPU ${gpu_id} ..."
    echo "  Config: exp=watch/ablation/${exp_name}"
    echo "  Log:    ${log_file}"

    CUDA_VISIBLE_DEVICES=${gpu_id} nohup ${PYTHON} ${SCRIPT} \
        exp=watch/ablation/${exp_name} \
        pl_trainer.devices=1 \
        > "${log_file}" 2>&1 &

    echo "  PID: $!"
    echo ""
}

run_single() {
    local gpu_id=$1
    local exp_name=$2
    local log_file="outputs/watch_ablation/${exp_name}/train_watch.log"

    mkdir -p "outputs/watch_ablation/${exp_name}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${exp_name} on GPU ${gpu_id} ..."
    echo "  Log: ${log_file}"

    CUDA_VISIBLE_DEVICES=${gpu_id} ${PYTHON} ${SCRIPT} \
        exp=watch/ablation/${exp_name} \
        pl_trainer.devices=1 \
        2>&1 | tee "${log_file}"
}

# ===== Main =====
MODE=${1:-all}

if [ "$MODE" = "single" ]; then
    GPU_ID=${2:?  "Usage: $0 single GPU_ID EXP_NAME"}
    EXP_NAME=${3:? "Usage: $0 single GPU_ID EXP_NAME"}
    run_single "$GPU_ID" "$EXP_NAME"
    exit 0
fi

echo "============================================================"
echo " WATCH Ablation & Improvement Experiments"
echo " Mode: ${MODE}"
echo " Time: $(date)"
echo "============================================================"
echo ""

# Check GPU availability
echo "GPU Status:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo ""

if [ "$MODE" = "all" ] || [ "$MODE" = "abl" ]; then
    echo ">>> Launching ${#ABL_EXPERIMENTS[@]} Ablation Experiments..."
    for exp_line in "${ABL_EXPERIMENTS[@]}"; do
        read -r gpu_id exp_name desc <<< "$exp_line"
        run_experiment "$gpu_id" "$exp_name" "$desc"
    done
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "imp" ]; then
    echo ">>> Launching ${#IMP_EXPERIMENTS[@]} Improvement Experiments..."
    for exp_line in "${IMP_EXPERIMENTS[@]}"; do
        read -r gpu_id exp_name desc <<< "$exp_line"
        run_experiment "$gpu_id" "$exp_name" "$desc"
    done
fi

echo "============================================================"
echo " All experiments launched! Use 'nvidia-smi' to monitor."
echo " Logs are in: outputs/watch_ablation/<exp_name>/train_watch.log"
echo ""
echo " To monitor a specific experiment:"
echo "   tail -f outputs/watch_ablation/abl_no_camtraj/train_watch.log"
echo ""
echo " To kill all experiments:"
echo "   pkill -f 'scripts/main.py.*watch/ablation'"
echo "============================================================"
