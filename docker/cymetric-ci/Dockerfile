
FROM cyclus/cyclus

RUN apt-get install -y --force-yes python3-pip
RUN pip3 install cython numpy
RUN pip3 install tables
RUN pip3 install jinja2
RUN pip3 install pandas
RUN pip3 install numexpr argparse

COPY . /cymetric
WORKDIR /cymetric
RUN python setup.py install
RUN export PATH="~/local/bin:${PATH}"
RUN export LD_LIBRARY_PATH="~/local/lib:${LD_LIBRARY_PATH}"
RUN nosetests3 -w tests

