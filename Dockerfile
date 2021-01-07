FROM nvidia/cuda:10.2-devel-ubuntu18.04

USER root
WORKDIR /docker

# necessary for tkinter. otherwise it freezes
ENV TZ=Europe/Brussels
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#VOLUME volume

# install some packages with apt-get
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.6-dev \
    libpython3-dev \
    python3-pip \
    python3-tk \
    python3-setuptools \
    libgtk-3.0 \
    libgl1-mesa-glx \
    libyaml-dev \
    git \
    vim \
    sudo && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# make a symlink to run python3 as python
RUN ln -s /usr/bin/python3 /usr/bin/python & \
    ln -s /usr/bin/pip3 /usr/bin/pip

RUN git config --global http.sslverify false

# install libraries with pip
RUN pip install --upgrade pip && \
    pip install --no-cache-dir cython==0.29.21 \
                                torch==1.6.0 \
                                torchvision==0.7.0 \
                                scikit-build==0.11.1 \
                                cmake==3.18.4.post1 \
                                opencv-python==4.5.1.48 \
                                tensorboard==2.4.0 \
                                filterpy==1.4.5 \
                                albumentations==0.4.6 \
                                scikit-image==0.17.2 \ 
                                lap==0.4.0 \
                                tqdm==4.55.1 \
                                timm==0.1.20 \
                                terminaltables==3.1.0 \
                                tensorboardX==2.1 \
                                six==1.15.0 \
                                scipy==1.1.0 \
                                PyYAML==5.3.1 \
                                natsort==7.1.0 \
                                munkres==1.1.4 \
                                matplotlib==3.3.3 \
                                easydict==1.9 \
                                websocket-client==0.57.0 \
                                tornado==6.1 \
                                torchfile==0.1.0 \
                                requests==2.25.1 \
                                pyzmq==20.0.0 \
                                jsonpatch==1.28

# Clone the repo and install alphapose
RUN git clone https://github.com/maxmarkov/track_and_count && \
    export PATH=/usr/local/cuda/bin/:$PATH && \
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH && \
    cd track_and_count/libraries/alphapose && \
    python setup.py build develop

RUN cd track_and_count && \
    ./yolov5/weights/download_weights.sh && \
    ./deep_sort/deep_sort/deep/checkpoint/download_weights.sh

# TO DO: 
# Add automatic weights downloader
