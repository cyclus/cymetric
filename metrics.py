#! /usr/bin/env python
from argparse import ArgumentParser
import sqlite3

from queries import activity
from queries import waste
from queries import decayheat
from queries import radiotoxicity
from queries import massflow

metrics = {
    'activity': activity.activity,
    'relative-activity': radiotoxicity.rel_activity,
    'radiotoxicity': radiotoxicity.radiotoxicity,
    'waste-mass': waste.wastemass,
    'tot-nuc-mass': waste.nucmass,
    'waste-composition': waste.wastecomp,
    'decay-heat': decayheat.decayheat,
    'end-decay-heat': decayheat.enddecayheat,
    'fuel-mass': massflow.fuelmass
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
