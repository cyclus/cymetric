import sqlite3

def wastemass(c):
    """Lists mass of waste (in kg) transacted with respect to time.

    Args:
        c: connection cursor to sqlite database.
    """
    sql = """SELECT resources.TimeCreated, resources.Quantity 
             FROM resources 
             INNER JOIN transactions ON transactions.ResourceId =
resources.ResourceId 
             WHERE transactions.ReceiverId=14 
             ORDER BY resources.TimeCreated;"""

    cur = c.execute(sql)
    results = cur.fetchall()

    return results

def wastecomp(c):
    """Lists mass of each nuclide (in kg) with respect to time-step.

    Args:
        c: connection cursor to sqlite database.
    """
    sql = """SELECT transactions.Time, compositions.NucId, 
                    compositions.MassFrac*resources.Quantity
             FROM compositions
             INNER JOIN transactions ON resources.ResourceId =
transactions.ResourceId
             INNER JOIN resources ON compositions.QualId = resources.QualId
             WHERE transactions.ReceiverId=14
             ORDER BY transactions.Time;"""

    cur = c.execute(sql)
    results = cur.fetchall()

    return results

def nucmass(c):
    """Lists mass of each nuclide (in kg) at end of simulation.

    Args:
        c: connection cursor to sqlite database.
    """

    # Retrieve list of masses for each nuclide and time step
    allwaste = wastecomp(c)

    dict_waste = {}

    # Get only one nuclide per entry, add masses of same nuclide
    for time_step, nuc, mass in allwaste:
        if nuc in dict_waste.keys():
            dict_waste[nuc] += mass
        else:
            dict_waste[nuc] = mass

    # Put back into list of tuples & sort by nuclide
    nucmass = dict_waste.items()
    nucmass.sort()

    return nucmass
