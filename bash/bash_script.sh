export PIP_DOWNLOAD_CACHE=/om2/user/ehoseini/.pip/cache
export XDG_CACHE_HOME=/om2/user/ehoseini/st
export SINGULARITY_CACHEDIR=/om2/user/ehoseini/st
export OMPI_MCA_opal_cuda_support=true
export CUDA_HOME=/cm/shared/openmind/cuda/11.1/
module load openmind/cuda/11.1
module load openmind/gcc/5.3.0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/om/user/ehoseini/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
eval "$__conda_setup"
else
if [ -f "/om/user/ehoseini/miniconda3/etc/profile.d/conda.sh" ]; then
. "/om/user/ehoseini/miniconda3/etc/profile.d/conda.sh"
else
export PATH="/om/user/ehoseini/miniconda3./bin:$PATH"
fi
fi
unset __conda_setup
# <<< conda initialize <<<
