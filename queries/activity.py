import sqlite3
from pyne import data


CONV = 1000*6.022e23

def activity(c, ):
    """Lists activities of all nuclides with respect to time for all facilities.

    Args:
        c: connection cursor to sqlite database.
    """
    sql = """SELECT Resources.TimeCreated, Compositions.NucID, 
                    Compositions.MassFrac*Resources.Quantity*""" 
                    + CONV*data.decay_const("""Compositions.NucID""")/data.atomic_mass("""Compositions.NucID""") + 
                    """
             FROM Resources
             INNER JOIN Compositions ON Resources.StateID = Compositions.StateID
             GROUP BY Resources.TimeCreated, Compositions.NucID
             ORDER BY Resources.TimeCreated;"""
    return c.execute(sql)
    
def activity(c):
    """Lists activities of all nuclides with respect to time for all HWRs.

    Args:
        c: connection cursor to sqlite database.
    """
    sql = """SELECT Resources.TimeCreated, Compositions.NucID, Compositions.MassFrac*Resources.Quantity*""" 
                    + CONV*data.decay_const("""Compositions.NucID""")/data.atomic_mass("""Compositions.NucID""") + 
                    """                    
             FROM Resources
             INNER JOIN Compositions ON Resources.StateID = Compositions.StateID
             INNER JOIN Transactions ON  Resources.ResourceID = Transactions.ResourceID
             WHERE Transactions.SenderID=21
             GROUP BY Resources.TimeCreated, Compositions.NucID
             ORDER BY Resources.TimeCreated;"""
    return c.execute(sql)
