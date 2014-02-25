#! /usr/bin/env python
from argparse import ArgumentParser
import sqlite3

from queries import activity
from queries import waste

metrics = {
    'activity': activity.activity,
    'waste-mass': waste.waste_mass,
    }

def main():
    """Calculates nuclear fuel cycle metrics from cyclus output sql database.
    """
    # cli
    parser = ArgumentParser()
    parser.add_argument('metric', help='metric name to compute')
    parser.add_argument('filename', help='file to execute on.')
    ns = parser.parse_args()

    # Open cyclus output
    conn = sqlite3.connect(ns.filename)
    c = conn.cursor()

    # Call functions from query and do analyses
    f = metrics[ns.metric]
    result = f(c)
    print(result)

if __name__ == "__main__":
    main()
