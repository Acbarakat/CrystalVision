FROM nvcr.io/nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04


# # TensorRT
# wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/local_repos/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-11.8_1.0-1_amd64.deb
# dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.0.0.11-ga-20191216_1-1_amd64.deb
# cp /var/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-11.8/nv-tensorrt-local-0628887B-keyring.gpg /usr/share/keyrings/

# # Install All The Things
# apt-get update
# apt-get -y install cuda-toolkit-12-3 cudnn-cuda-12 nvidia-cuda-toolkit tensorrt-dev libnvinfer-dev python3-libnvinfer-dev


# # Add to .bashrc
# export CUDA_HOME=/usr/local/cuda
# export DYLD_LIBRARY_PATH=$CUDA_HOME/lib64:$DYLD_LIBRARY_PATH
# export PATH=$CUDA_HOME/bin:$PATH
# export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
# export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# export LD_RUN_PATH=$CUDA_HOME/lib64:$LD_RUN_PATH


# # IPPI
# wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/046b1402-c5b8-4753-9500-33ffb665123f/l_ipp_oneapi_p_2021.10.1.16.sh
# ./l_ipp_oneapi_p_2021.10.1.16.sh --silent

# # Add to .bashrc
# source /home/acbaraka/intel/oneapi/setvars.sh