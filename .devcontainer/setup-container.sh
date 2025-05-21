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
