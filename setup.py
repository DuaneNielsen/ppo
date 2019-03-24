#!/usr/bin/env python
from setuptools import setup

setup(name='ppo',
      version='0.0.1',
      install_requires=['gym>=0.2.3',
                        'torch',
                        'numpy',
                        'py3nvml',
                        'tensorboardX',
                        'gym-duane',
                        'redis',
                        'opencv-python',
                        'torchvision'],
      extras_require={
          'dev': [
              'pytest',
              'tox'
          ]
      }
      )
