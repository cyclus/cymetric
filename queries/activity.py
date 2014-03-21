import sqlite3
import math
from pyne import data

def query(c):
    """Lists activities of all nuclides with respect to time for all facilities.

    Args:
        c: connection cursor to sqlite database.
    """

    # SQL query returns a table with the nuclides and their masses at each timestep
    sql = ("SELECT Resources.TimeCreated, Compositions.NucID," 
           "Compositions.MassFrac*Resources.Quantity ")
    sql += ("FROM Resources "
            "INNER JOIN Compositions ON Resources.StateID = Compositions.StateID "
            "GROUP BY Resources.TimeCreated, Compositions.NucID "
            "ORDER BY Resources.TimeCreated;")
    cur = c.execute(sql)
    results = cur.fetchall()

    # Gives avogadro's number with a kg to g conversion 
    CONV = 1000*6.022e23

    activities = []

    # Calculates activities (/s) of each nuclide at each timestep
    for time_step, nuc, mass in results:
        act = CONV * mass * data.decay_const(nuc) / data.atomic_mass(nuc)
        row = (time_step, nuc, act)
        activities.append(row)

    return activities

def activity(c):
    """Lists activities of all nuclides at a given time (in years) from the end of the sim.

    Args:
        c: connection cursor to sqlite database.
    """

    activities = query(c)

    # Conversion of time from months to seconds
    MCONV = 3.16e7 / 12

    # Conversion of years to seconds
    YCONV = 3.16e7

    dict_acts = {}

    t = input("Enter a time in years: ")

    sim_time = activities[-1][0]
    time = sim_time * MCONV + t * YCONV

    # Get only one nuclide per entry, add activities
    for time_step, nuc, act in activities:
        sec = time - time_step * MCONV
        act = act * math.exp(-sec * data.decay_const(nuc))
        if nuc in dict_acts.keys():
            dict_acts[nuc] += act
        else:
            dict_acts[nuc] = act

    # Put back into list of tuples & sort by nuclide
    acts = dict_acts.items()
    acts.sort()

    return acts
