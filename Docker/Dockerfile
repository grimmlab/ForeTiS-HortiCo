FROM nvidia/cuda:11.1.1-base-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y python3 && apt-get install -y python3-pip  &&  \
    apt-get install -y r-base && apt-get install -y vim && apt-get install -y git
RUN mkdir /configfiles
COPY docker_requirements.txt /configfiles
RUN pip3 install -r /configfiles/docker_requirements.txt

