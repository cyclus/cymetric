ARG pkg_mgr=apt
ARG ubuntu_version=22.04
ARG cycamore_tag=stable

FROM ghcr.io/cyclus/cycamore_${ubuntu_version}_${pkg_mgr}/cycamore:${cycamore_tag} as cymetric

COPY . /cymetric
WORKDIR /cymetric
RUN python -m pip install --target $(python -m site --user-site) .

FROM cymetric as cymetric-pytest
RUN cd tests && python -m pytest