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
    ns = parser.parse_args()
    return ns


def main():
    """CLI entry point."""
    ns = parse_args()
    db = dbopen(ns.db)
    if ns.exec_code is not None:
        evaler = Evaluator(db)
        gbl = {}
        loc = ExecutionContext(evaler=evaler)
        exec(ns.exec_code, glb, loc)


if __name__ == '__main__': 
    main()
