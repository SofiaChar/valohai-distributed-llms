# Use the Python 3.9 base image
FROM nvidia/cuda:12.2.0-base-ubuntu20.04

# Update the package list and install necessary dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.9 python3.9-dev python3.9-distutils python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Create a symbolic link for python3.9 to python3
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Upgrade pip to the latest version
RUN python -m pip install --upgrade pip
# Install required libraries
RUN pip install torch sentence_transformers chromadb stuff transformers matplotlib datasets nltk py7zr valohai-utils accelerate -U

# Set the working directory
WORKDIR /app

# Copy your application files to the container (if needed)
# COPY . /app

# Specify the default command to be executed when the container starts
# CMD [ "python", "your_script.py" ]
