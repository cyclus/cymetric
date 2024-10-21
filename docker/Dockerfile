ARG pkg_mgr=apt
ARG ubuntu_version=22.04
ARG cycamore_tag=latest

FROM ghcr.io/cyclus/cycamore_${ubuntu_version}_${pkg_mgr}/cycamore:${cycamore_tag} AS cycamore-base

FROM cycamore-base AS apt-deps
RUN apt install -y python3-matplotlib

FROM cycamore-base AS conda-deps
RUN mamba install -y matplotlib

FROM ${pkg_mgr}-deps AS cymetric
COPY . /cymetric
WORKDIR /cymetric
RUN python3 -m pip install --target $(python3 -m site --user-site) --no-deps .

FROM cymetric AS cymetric-pytest
RUN cd tests && python3 -m pytest
