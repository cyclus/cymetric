import sqlite3

def wastemass(c):
    """Lists mass of waste (in kg) transacted with respect to time.

    Args:
        c: connection cursor to sqlite database.
    """
    sql = """SELECT Resources.TimeCreated, Resources.Quantity 
             FROM Resources 
             INNER JOIN Transactions ON Transactions.ResourceID = Resources.ResourceID 
             WHERE Transactions.ReceiverID=25 
             ORDER BY Resources.TimeCreated;"""

    cur = c.execute(sql)
    results = cur.fetchall()

    return results

def wastecomp(c):
    """Lists mass of each nuclide (in kg) with respect to time-step.

    Args:
        c: connection cursor to sqlite database.
    """
    sql = """SELECT Transactions.Time, Compositions.NucID, 
                    Compositions.MassFrac*Resources.Quantity
             FROM Compositions
             INNER JOIN Transactions ON  Resources.ResourceID = Transactions.ResourceID
             INNER JOIN Resources ON Compositions.StateID = Resources.StateID
             WHERE Transactions.ReceiverID=25
             ORDER BY Transactions.Time;"""

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
