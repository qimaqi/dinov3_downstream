#!/bin/bash
#SBATCH --job-name=train_flexcit_linear_prob_f0_custom_pretrain_fomomri_fomo100k
#SBATCH --output=sbatch_log/train_flexcit_linear_prob_f0_custom_pretrain_fomomri_fomo100k_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a6000:1

##SBATCH --nodelist=octopus02
#SBATCH --cpus-per-task=8
#SBATCH --mem 100GB
#SBATCH --mail-type=BEGIN,END,FAIL


# Load any necessary modules
# source /usr/bmicnas03/data-biwi-01/qimaqi_data/data/miniconda3/etc/profile.d/conda.sh
# conda activate flexict

# scratch env debug
export JOB_SCRATCH=/scratch/qimaqi/${SLURM_JOB_ID}
export CONDA_PACK_IGNORE_EDITABLE=1
export CONDA_PACK_IGNORE_MISSING=1

source "$HOME/scripts/prepare_env_in_scratch.sh" flexict
source "$HOME/scripts/prepare_flexict_sources_in_scratch.sh"



export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export PATH=//usr/bmicnas03/data-biwi-01/qimaqi_data/data/schusch_archiv/install_gcc:$PATH
export CC=//usr/bmicnas03/data-biwi-01/qimaqi_data/data/schusch_archiv/install_gcc/bin/gcc-11.3.0
export CXX=//usr/bmicnas03/data-biwi-01/qimaqi_data/data/schusch_archiv/install_gcc/bin/g++-11.3.0

# export FLEXICT_2D_CHECKPOINT=/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/FlexiCT/ckpts/2D_final_model.pth
# export FLEXICT_3D_CHECKPOINT=/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/FlexiCT/ckpts/3D_final_model.pth


# export FLEXICT_2D_CHECKPOINT=/usr/bmicnas03/data-biwi-01/qimaqi_data/data/medical_journal/flexcit_outputs/leomed/pretrain_fomo_10k_pretrained_flexcit_base_g8_e200_p8_mri//2D_final_model.pth
# export FLEXICT_3D_CHECKPOINT=/usr/bmicnas03/data-biwi-01/qimaqi_data/data/medical_journal/flexcit_outputs/leomed/pretrain_fomo_10k_pretrained_flexcit_base_g8_e200_p8_mri//3D_final_model.pth

export FLEXICT_2D_CHECKPOINT=/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/FlexiCT_downstream/FlexiCT/ckpts/pretrain_fomo_100k_pretrained_flexcit_base_g8_e200_p8_mri_gram/2D_final_model_fomo100k_gram.pth
export FLEXICT_3D_CHECKPOINT=/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/FlexiCT_downstream/FlexiCT/ckpts/pretrain_fomo_100k_pretrained_flexcit_base_g8_e200_p8_mri_gram/2D_final_model_fomo100k_gram.pth




export nnUNet_raw="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw"
export nnUNet_preprocessed="/usr/bmicnas03/data-biwi-01/qimaqi_data/data/medical_journal/meddinov3_preprocessed"
export nnUNet_results="/usr/bmicnas03/data-biwi-01/qimaqi_data/data/medical_journal/flexict_results_fomo100k/" 
cd /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/FlexiCT_downstream/FlexiCT/downstream/segmentation
export DATASET_ID="${DATASET_ID:-69}"
export CONFIG="${CONFIG:-2d}"
export FOLD="${FOLD:-0}"
export TRAINER="${TRAINER:-flexict_patch8_onescale_linear_Trainer}"
PRETRAINED_WEIGHTS="" ./run_train.sh "$@"




export DATASET_ID="${DATASET_ID:-71}"
export CONFIG="${CONFIG:-2d}"
export FOLD="${FOLD:-0}"
export TRAINER="${TRAINER:-flexict_patch8_onescale_linear_Trainer}"
PRETRAINED_WEIGHTS="" ./run_train.sh "$@"
