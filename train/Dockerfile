# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
# refer to the above page to pull latest Pytorch image

# docker image region us-west-2
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-cpu-py39-ubuntu20.04-sagemaker AS hfcpu

# version args
ARG TRANSFORMERS_VERSION=4.26.0
ARG DATASETS_VERSION=2.12.0
ARG PYTHON=python3

# AWS packages
ARG PT_TORCHAUDIO_URL=https://download.pytorch.org/whl/cu117/torchaudio-0.13.1%2Bcu117-cp39-cp39-linux_x86_64.whl

# install Hugging Face libraries and its dependencies
RUN pip install --no-cache-dir \
	transformers[sklearn,sentencepiece,audio,vision]==${TRANSFORMERS_VERSION} \
	datasets==${DATASETS_VERSION} \
	$PT_TORCHAUDIO_URL

RUN apt-get update \
 # TODO: Remove upgrade statements once packages are updated in base image
 && apt-get -y upgrade --only-upgrade systemd openssl cryptsetup \
 && apt install -y git-lfs \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN HOME_DIR=/root \
 && curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip \
 && unzip ${HOME_DIR}/oss_compliance.zip -d ${HOME_DIR}/ \
 && cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance \
 && chmod +x /usr/local/bin/testOSSCompliance \
 && chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh \
 && ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} ${PYTHON} \
 && rm -rf ${HOME_DIR}/oss_compliance*

# Install additional requirements
COPY src/requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt

FROM hfcpu AS devcontainer

COPY requirements-dev.txt requirements-dev.txt
RUN python -m pip install -r requirements-dev.txt

RUN apt-get update && apt-get install -y xz-utils

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
