#!/bin/bash
#SBATCH --job-name=mini
#SBATCH --time=2-12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=120G
#SBATCH -w node105
#SBATCH –gres=gpu:A100:4
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu
#SBATCH --partition=evlab

. ~/.bash_profile
conda activate gpt_neox
echo $(which python)
cd /om/user/ehoseini/gpt-neox/

python ./deepy.py train.py -d configs gpt2.yml miniBERTA_1m_v2_setup.yml miniBERTa_1m_v2_gpt2_logging_setup.yml



####SBATCH --gres=gpu:RTXA6000:1