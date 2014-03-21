import sqlite3

def waste_mass(c):
    """Lists mass of waste transacted with respect to time.

    Args:
        c: connection cursor to sqlite database.
    """
    sql = """SELECT Resources.TimeCreated, Resources.Quantity, Resources.units 
             FROM Resources 
             INNER JOIN Transactions ON Transactions.ResourceID = Resources.ResourceID 
             WHERE Transactions.ReceiverID=25 
             ORDER BY Resources.TimeCreated;"""

    cur = c.execute(sql)
    results = cur.fetchall()

    return results

def end_waste_comp(c):
    """Lists total mass of waste in facilty with respect to isotope at end of simulation.

    Args:
        c: connection cursor to sqlite database.
        t: time.
    """
    sql = """SELECT Compositions.NucID, Compositions.MassFrac*Resources.Quantity, Resources.units
             FROM Compositions
             INNER JOIN Transactions ON  Resources.ResourceID = Transactions.ResourceID
             INNER JOIN Resources ON Compositions.StateID = Resources.StateID
             WHERE Transactions.ReceiverID=25
             GROUP BY Compositions.NucID
             ORDER BY Compositions.NucID;"""

    cur = c.execute(sql)
    results = cur.fetchall()

    return results
