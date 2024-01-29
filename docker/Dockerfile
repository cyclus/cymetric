ARG pkg_mgr=apt
ARG ubuntu_version=22.04
ARG cycamore_tag=stable

FROM ghcr.io/cyclus/cycamore_${ubuntu_version}_${pkg_mgr}/cycamore:${cycamore_tag} as cycamore

FROM cycamore as apt-deps
RUN apt update --fix-missing && apt install -y python3-pip python3-pytest

FROM cycamore as conda-deps
RUN mamba install -y pip pytest

FROM ${pkg_mgr}-deps as cymetric
COPY . /cymetric
WORKDIR /cymetric
RUN python -m pip install --target $(python -m site --user-site) .

FROM cymetric as cymetric-pytest
RUN cd tests && python -m pytest