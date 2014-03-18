import sqlite3

def waste_mass(c):
    """Lists mass of waste transacted with respect to time.

    Args:
        c: connection cursor to sqlite database.
    """
    sql = """SELECT Transactions.ReceiverID, Transactions.ResourceID, 
                    Resources.TimeCreated, Resources.Quantity, Resources.units 
             FROM Transactions 
             INNER JOIN Resources ON Transactions.ResourceID = Resources.ResourceID 
             WHERE Transactions.ReceiverID=23 
             ORDER BY Resources.TimeCreated;"""
    return c.execute(sql)

def total_waste_content(c):
    """Lists total mass of waste in facilty with respect to isotope at end of simulation.

    Args:
        c: connection cursor to sqlite database.
        t: time.
    """
    sql = """SELECT Compositions.NucID, Compositions.MassFrac*Resources.Quantity, Resources.units
             FROM Compositions
             INNER JOIN Transactions ON  Resources.ResourceID = Transactions.ResourceID
             INNER JOIN Resources ON Compositions.StateID = Resources.StateID
             WHERE Transactions.ReceiverID=23
             GROUP BY Compositions.NucID
             ORDER BY Compositions.NucID;"""
    return c.execute(sql)

