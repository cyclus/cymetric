#!/bin/bash

if [ $# -ne 1 ]; then 
  echo "illegal number of parameters"
  exit 1
fi

if [ ! ${1} = "stable" ] && [ ! ${1} = "latest"  ] ; then
  echo "Version must be \"stable\" or \"latest\" !"
  exit 1
fi

rm Dockerfile
sed -e s/CYCAMORE_VERSION/${1}/g docker/cymetric-deps/Dockerfile_template > Dockerfile

#docker build -t cyclus/cymetric-deps:${1} docker/cymetric-deps
#docker login -e $DOCKER_EMAIL -u $DOCKER_USER -p $DOCKER_PASS
#docker push cyclus/cymetric-deps:${1}

