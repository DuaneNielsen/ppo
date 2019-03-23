FROM ubuntu:18.04
ADD setup.py /
RUN apt update
RUN apt install -y keyboard-configuration
RUN apt install -y python-setuptools
RUN apt install -y python-pip
RUN apt install -y python3-dev
RUN apt install -y python3-pip
RUN apt install -y python-pyglet
RUN apt install -y python3-opengl
RUN apt install -y libjpeg-dev
RUN apt install -y libboost-all-dev
RUN apt install -y libsdl2-dev
RUN apt install -y libosmesa6-dev
RUN apt install -y patchelf
RUN apt install -y ffmpeg
RUN apt install -y xvfb
RUN apt install -y wget
RUN apt install -y unzip
RUN pip3 install -e .
RUN apt install -y git
RUN pip3 install git+https://github.com/Kojoley/atari-py.git
RUN apt install -y swig
RUN pip3 install gym[atari,box2d,classic_control]