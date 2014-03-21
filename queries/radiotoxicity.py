import sqlite3
from pyne import data

import activity

def radiotoxicity(c):
    """Lists radiotoxicity of all nuclides with respect to time for all facilities.

    Args:
        c: connection cursor to sqlite database.
    """

    # fetches list of activities for all nuclides at each time step
    activities = activity(c)

    
