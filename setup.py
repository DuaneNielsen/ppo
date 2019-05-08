#!/usr/bin/env python
from setuptools import setup

setup(name='ppo',
      version='0.0.1',
      install_requires=['torch',
                        'gym[classic_control]>=0.2.3', 'gym[box2d]>=0.2.3', 'gym[atari]>=0.2.3',
                        'numpy',
                        'py3nvml',
                        'tensorboardX',
                        'gym-duane',
                        'redis',
                        'opencv-python',
                        'torchvision',
                        'psycopg2-binary',
                        'PySimpleGUI',
                        'peewee'],
      extras_require={
          'dev': [
              'pytest',
              'tox'
          ]
      }
      )
