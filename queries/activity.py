import sqlite3
from pyne import data

CONV = 1000*6.022e23

def activity(c):
    """Lists activities of all nuclides with respect to time for all facilities.

    Args:
        c: connection cursor to sqlite database.
    """
    sql = ("SELECT Resources.TimeCreated, Compositions.NucID," 
           "Compositions.MassFrac*Resources.Quantity ")
    sql += ("FROM Resources "
            "INNER JOIN Compositions ON Resources.StateID = Compositions.StateID "
            "GROUP BY Resources.TimeCreated, Compositions.NucID "
            "ORDER BY Resources.TimeCreated;")
    cur = c.execute(sql)
    results = cur.fetchall()
    activities = []
    for time_step, nuc, mass in results:
        act = CONV * data.decay_const(nuc) / data.atomic_mass(nuc)
        row = (time_step, nuc, act)
        activities.append(row)
    return activities
