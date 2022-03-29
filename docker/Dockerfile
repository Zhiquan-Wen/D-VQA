FROM nvidia/cuda:10.0-runtime-ubuntu18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH \
    TZ=Asia/Shanghai

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    GIT_CLONE="git clone --depth 10" && \
    cat /etc/apt/sources.list && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL tzdata git wget vim htop rsync psmisc unzip python3.6 python3-pip && \
    apt-get clean

RUN ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime && \
    echo ${TZ} > /etc/timezone && \
    dpkg-reconfigure --frontend noninteractive tzdata

COPY .condarc /root/.condarc

RUN PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade" && \
    $PIP_INSTALL setuptools && \
    $PIP_INSTALL tqdm==4.60.0 numpy==1.19.5 requests==2.25.1 h5py==3.1.0 botocore==1.20.47 torch==1.1.0 boto3==1.17.47 Pillow==8.1.2 && \
    echo 'PIP Install Done!'