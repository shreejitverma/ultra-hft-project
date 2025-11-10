# Use an x86-64 (amd64) base image, even on ARM
FROM --platform=linux/amd64 ubuntu:22.04

# Set non-interactive for installs
ENV DEBIAN_FRONTEND=noninteractive

# Install all build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cmake \
    git \
    libgtest-dev \
    libbenchmark-dev \
    pkg-config

# Set up a working directory
WORKDIR /app

# Add a non-root user
RUN useradd -ms /bin/bash builder
USER builder
WORKDIR /home/builder/project

# Default command
CMD ["/bin/bash"]