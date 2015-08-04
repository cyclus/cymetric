
* ``cymetric-ci`` is the dockerfile used for running cymetric on a continuous
  integration service.  This dockerfile assumes that the current working
  directory is a cymetric repository - and that version of cymetric is copied
  into the docker container and used for the build.  The dockerfile in the
  cycamore repository root is a symbolic link to this dockerfile.  This
  dockerfile uses the base image ``cyclus/cycamore`` from the docker hub
  repository.

