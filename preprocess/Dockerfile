FROM python:3.10-buster AS base

COPY src/requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt

ENV PYTHONUNBUFFERED=TRUE


FROM base AS devcontainer

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
