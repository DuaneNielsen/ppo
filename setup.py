#!/usr/bin/env python
from setuptools import setup

setup(name='ppo',
      version='0.0.1',
      install_requires=['git+https://github.com/Kojoley/atari-py.git',
                        'https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl'
                        'gym[classic_control]>=0.2.3', 'gym[box2d]>=0.2.3', 'gym[atari]>=0.2.3',
                        'numpy',
                        'py3nvml',
                        'tensorboardX',
                        'gym-duane',
                        'redis',
                        'opencv-python',
                        'torchvision',
                        'psycopg2-binary',
                        'PySimpleGUI'],
      extras_require={
          'dev': [
              'pytest',
              'tox'
          ]
      }
      )
