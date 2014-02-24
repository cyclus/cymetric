#! /usr/bin/env python

import sqlite3
from pyne import data

import query


def main():
    """Calculates nuclear fuel cycle metrics from cyclus output sql database.
    """
# Open cyclus output
conn = sqlite3.connect("./cylcus.sqlite")
c = conn.cursor()

nuclide = "010030000"
dc = data.decay_const(nuclide)

# Call functions from query and do analyses

if __name__ == "__main__":
    main()
