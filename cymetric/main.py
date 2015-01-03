"""The CLI for cymetric."""
from __future__ import unicode_literals, print_function
import os
import sys
from argparse import ArgumentParser, Namespace

from cymetric.tools import dbopen
from cymetric.evaluator import Evaluator
from cymetric.execution import ExecutionContext

def parse_args():
    """Parses the command line arguments."""
    parser =  ArgumentParser(description='Cyclus metric analysis tool.')
    parser.add_argument('db', help='path to the database')
    parser.add_argument('-e', dest='exec_code', help='execution string',
                        default=None)
    parser.add_argument('-l', dest='listing', action='store_true', 
                        help='lists the tables in the database', default=False)
    ns = parser.parse_args()
    return ns


def main():
    """CLI entry point."""
    ns = parse_args()
    db = dbopen(ns.db)
    if ns.listing:
        for tab in sorted(db.tables()):
            print(tab)
    if ns.exec_code is not None:
        evaler = Evaluator(db)
        glb = {}
        loc = ExecutionContext(evaler=evaler)
        exec(ns.exec_code, glb, loc)


if __name__ == '__main__': 
    main()
