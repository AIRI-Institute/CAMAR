# apt-get update
# apt-get upgrade -y
# # install Linux tools and Python 3
# apt-get install build-essential software-properties-common wget curl git ffmpeg libsm6 libxext6 vim
# # update CUDA Linux GPG repository key
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
# dpkg -i cuda-keyring_1.0-1_all.deb
# rm cuda-keyring_1.0-1_all.deb
# # install cuDNN
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
# mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
# apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
# add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" -y
# apt-get update
# apt-get install libcudnn8=8.9.0.*-1+cuda11.8
# apt-get install libcudnn8-dev=8.9.0.*-1+cuda11.8
# # install recommended packages
# apt-get install zlib1g g++ freeglut3-dev \
#     libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev -y
# # clean up
# apt-get autoremove -y
# apt-get clean
# rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*=


apt update
apt install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    curl \
    vim

apt-get autoremove -y 
apt-get clean 
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*=
apt update
apt-get install -y cmake g++ wget

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
