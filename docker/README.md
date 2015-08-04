
* ``cymetric-ci`` is the dockerfile used for running cymetric on a continuous
  integration service.  This dockerfile assumes that the current working
  directory is a cymetric repository - and that version of cymetric is copied
  into the docker container and used for the build.  The dockerfile in the
  cycamore repository root is a symbolic link to this dockerfile.  This
  dockerfile uses the base image ``cyclus/cycamore`` from the docker hub
  repository.

* ``cymetric-deps`` builds all cymetric dependencies.  This is used as the
  base image for other dockerfiles that build cymetric and should be updated
  only occasionally as needed (i.e. whenever we do a new release of
  cyclus+cycamore) and pushed up to the docker hub ``cyclus/cyclus-deps``
  repository:

  ```
  cd cymetric-deps
  docker build -t cyclus/cymetric-deps:X.X .
  docker tag cyclus/cymetric-deps:X.X cyclus/cymetric:latest
  docker push cyclus/cymetric-deps
  ```

