FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu14.04

LABEL maintainer "miyuu.osanai@kuno-corp.com"

RUN locale-gen en_US.UTF-8  
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install git make build-essential python-dev python-pip libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev curl
RUN git clone git://github.com/yyuu/pyenv.git .pyenv

ENV HOME  /
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN pyenv install 3.6.0
RUN pyenv global 3.6.0

RUN git clone https://3bc95143eb84df7ab068b0ec6620c23aa1645597:x-oauth-basic@github.com/macromilldev/MHS-AI /app/MHS-AI
RUN pip install -r /app/MHS-AI/requirements-pip-gpu.txt