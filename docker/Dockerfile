ARG pkg_mgr=apt
ARG ubuntu_version=22.04

FROM ghcr.io/cyclus/cycamore_${ubuntu_version}_${pkg_mgr}/cycamore as cymetric
ARG make_cores=2

COPY . /cymetric
WORKDIR /cymetric

RUN apt update && apt install -y python3-pip
RUN python -m pip install --target $(cyclus --install-path) .

FROM cymetric as cymetric-pytest
RUN apt update && apt install python3-pytest
RUN cd tests && python -m pytest