Installation procedure to prepare environment for running the scripts:

# I. Docker.

The repository already has a Dockerfile ready for build. It starts with nvidia CUDA image and sequentially installs all necessary libraries with apt-get and pip.
Most of the libraries are required for installation of alphapose code. Alphapose has to be build separately since it includes a code written on C/C++.  

1. Update your daemon.json

sudo vi /etc/docker/daemon.json

{"runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
                  }
             },
             "default-runtime": "nvidia"
}

2. Restart your docker to enable CUDA for building alphapose with docker build

sudo systemctl restart docker

3. Build a docker image with the filename track

docker build --network=host -t track . 

4. Create a docker volume with the filename vol_track to enable transfer of files between the host and containers

docker volume create vol_track

5. Run a container with an activated GPU and a mounted volume 

docker run --runtime=nvidia --net=host --env="DISPLAY" --privileged -it -u 0 -v vol_track:/docker/data/ --rm track

Code is ready to use!

**NB**: docker already downloads all necessary weights for yolo, deep_sort and alphapose. Alphapose is already compiled and ready for use.

# II. Anaconda/pip (no alphapose).

1. Create a new conda environment (optional)

conda create --name env_track python=3.6

2. Install all required libraries with pip 

pip3 install -r requirements.txt

3. Download weights

./yolov5/weights/download_weights.sh
./deep_sort/deep_sort/deep/checkpoint/download_weights.sh

Code is ready to use!

# III. Alphapose installation with conda:

1. Create a new conda environment (optional)

conda create --name env_track python=3.6

2. Install all required libraries with pip

pip3 install -r requirements.txt

3. Export CUDA paths to your ~/.bashrc   

export PATH=/usr/local/cuda-10.0/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64/:$LD_LIBRARY_PATH

4. Install compilers with conda (optional, in my case g++ was absent):

conda install -c anaconda gxx_linux-64  

5. Create symbolic links for conda compilers. Here YOUR_PATH is your path to anaconda.

ln -s YOUR_PATH/anaconda3/envs/env_track/bin/x86_64-conda_cos6-linux-gnu-gcc YOUR_PATH/anaconda3/envs/env_alpha/bin/gcc
ln -s YOUR_PATH/anaconda3/envs/env_track/bin/x86_64-conda_cos6-linux-gnu-g++ YOUR_PATH/anaconda3/envs/env_alpha/bin/g++

6. Run the installation of alphapose:

cd libraries/alphapose/
python3 setup.py build develop

7. Download the weights:

./download_weights.sh

Code is ready to use!

8. Test alphapose on demo images:

python3 scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/ --outdir examples/inference --vis_fast --save_img

or

./run.sh
