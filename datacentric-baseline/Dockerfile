FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1

RUN apt-get -y update && \
    apt-get -y install git && \
    groupadd -r algorithm && \
    useradd -m --no-log-init -r -g algorithm algorithm && \
    mkdir -p /opt/algorithm /input /output /output/images/automated-petct-lesion-segmentation  && \
    chown -R algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm weights/ /opt/algorithm/weights/
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm predict.py /opt/algorithm/

RUN python -m pip install --user -U pip && \
    python -m pip install --user -r requirements.txt

ENTRYPOINT ["python", "-m", "process", "$0", "$@"]
