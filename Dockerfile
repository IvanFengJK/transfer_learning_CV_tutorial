FROM nvcr.io/nvidia/pytorch:20.12-py3

ADD src /src

ADD data /data

COPY build /build
WORKDIR /build
RUN pip install docker-py --ignore-installed PyYAML
RUN pip install -r requirements.txt

CMD ["cd", "src"]