import sqlite3

def waste_mass(c):
    """Lists mass of waste transacted with respect to time.

    Args:
        c: connection cursor to sqlite database.
    """
    sql = """SELECT Transactions.ReceiverID, Transactions.ResourceID, 
                    Resources.Quantity, Resources.units, Resources.TimeCreated 
             FROM Transactions 
             INNER JOIN Resources ON Transactions.ResourceID = Resources.StateID 
             WHERE Transactions.ReceiverID=23 
             ORDER BY Resources.TimeCreated;"""
    return c.execute(sql)

def waste_content(c):
    """Lists total mass of waste in facilty with respect to isotope at end of simulation.

    Args:
        c: connection cursor to sqlite database.
    """
    sql = """SELECT Compositions.IsoID, Compositions.Quantity*Resources.Quantity, Resources.units
             FROM Compositions
             INNER JOIN Transactions ON  Compositions.ID = Transactions.ResourceID
             INNER JOIN Resources ON Compositions.ID = Resources.StateID
             WHERE Transactions.ReceiverID=23
             GROUP BY Compositions.IsoID
             ORDER BY Compositions.IsoID;"""
    return c.execute(sql)

