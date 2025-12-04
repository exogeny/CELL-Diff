# Using Weights & Biases (WandB) for experiment tracking (optional)
# export WANDB_RUN_NAME=       # (Optional) Name of the current run (visible in the WandB dashboard)
# export WANDB_API_KEY=        # (Optional) Your WandB API key to authenticate logging
# export WANDB_PROJECT=        # (Optional) WandB project name to group related experiments
export WANDB_DISABLED=true
export OMP_NUM_THREADS=32
# export CUDA_VISIBLE_DEVICES=2,3
export n_gpu=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set the output directory for training results
export output_dir=./train_dir_ns

# Set the path to the training dataset
export data_path=/mnt/data1/tensorflow_datasets/extracted/human_protein_atlas/ProteinGeneration

# Set the path to the pretrained VAE checkpoint
export vae_loadcheck_path=/ldap_shared/home/s_wt/CELL-Diff/pretrained_models/vae/hpa_pretrained.bin

bash scripts/cell_diff/pretrain_hpa.sh
