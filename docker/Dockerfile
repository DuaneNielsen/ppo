FROM ubuntu:18.04
RUN apt-get update
RUN apt-get install -y keyboard-configuration

# python 3
RUN apt-get install -y python-setuptools
RUN apt-get install -y python-pip
RUN apt-get install -y python3-dev
RUN apt-get install -y python3-pip

# i think this is needed for gym
RUN apt-get install -y python-pyglet
RUN apt-get install -y python3-opengl
RUN apt-get install -y libjpeg-dev
RUN apt-get install -y libboost-all-dev
RUN apt-get install -y libsdl2-dev
RUN apt-get install -y libosmesa6-dev
RUN apt-get install -y patchelf
RUN apt-get install -y ffmpeg
RUN apt-get install -y xvfb
RUN apt-get install -y wget
RUN apt-get install -y unzip
RUN apt-get install -y git

# debugging utils
RUN apt-get install -y iputils-ping

# required for AI gym
RUN pip3 install git+https://github.com/Kojoley/atari-py.git
RUN apt-get install -y swig
RUN pip3 install gym[box2d]
RUN pip3 install gym[classic_control]
RUN pip3 install gym[atari]

# ray requires the below env variables for python 3
ENV LC_ALL "C.UTF-8"
ENV LANG "C.UTF-8"
RUN pip3 install psutil
RUN pip3 install ray

# torch on linux with CUDA 10
RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torchvision

# the project
ADD https://api.github.com/repos/DuaneNielsen/ppo/git/refs/heads/master version.json
RUN git clone https://github.com/DuaneNielsen/ppo.git
RUN cd ppo && pip3 install -e .