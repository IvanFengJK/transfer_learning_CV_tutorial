# docker build -t pytorch_image_tl .
# docker run -it --entrypoint /bin/bash --shm-size 8G -v /home/elwin/Desktop/transfer_learning_CV_tutorial/data:/data pytorch_image_tl
FROM nvcr.io/nvidia/pytorch:20.12-py3

COPY build /build
WORKDIR /build
RUN pip install -r requirements.txt

WORKDIR /src