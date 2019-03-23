FROM duanes-base
ADD *.py /
RUN pip3 install -e .
CMD [ "python3", "./ppo_clip_discrete.py"]