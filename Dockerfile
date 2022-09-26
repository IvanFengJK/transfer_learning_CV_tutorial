# docker
# docker run -it --entrypoint /bin/bash --shm-size 8G -v /home/elwin/Desktop/transfer_learning_CV_tutorial/data:/data pytorch_image_tl
FROM nvcr.io/nvidia/pytorch:20.12-py3

ADD src /src

COPY build /build
WORKDIR /build
RUN pip install -r requirements.txt

WORKDIR /src