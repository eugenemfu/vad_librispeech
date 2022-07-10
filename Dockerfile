FROM ubuntu:20.04

RUN apt-get update 
RUN apt-get install -y --no-install-recommends python3-pip python-is-python3

WORKDIR /root

COPY requirements.txt requirements.txt
SHELL ["/bin/bash", "-c"]
RUN pip3 install -r requirements.txt

VOLUME /root/workdir
VOLUME /root/audios
VOLUME /root/alignments
VOLUME /root/noises
VOLUME /root/rirs
VOLUME /root/augmented_audios
VOLUME /root/features_labels
VOLUME /root/infer

WORKDIR /root/workdir

CMD bash
