ARG pkg_mgr=apt
ARG ubuntu_version=22.04

FROM ghcr.io/cyclus/cycamore_${ubuntu_version}_${pkg_mgr}/cycamore as cycamore

COPY . /cymetric
WORKDIR /cymetric

FROM cymetric as conda-deps
RUN mamba install -y pip pytest

FROM cymetric as apt-deps
RUN apt update && apt install -y python3-pip python3-pytest

FROM ${pkg_mgr}-deps as cymetric
RUN python -m pip install --target $(python -m site --user-site) .

FROM cymetric as cymetric-pytest
RUN cd tests && python -m pytest