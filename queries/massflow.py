import sqlite3

def fuelmass(c):
    """Lists mass (in kg) of enriched fuel transacted from with respect to nuclide and time.

    Args:
        c: connection cursor to sqlite database.
    """
    sql = """SELECT Resources.TimeCreated, Compositions.MassFrac*Resources.Quantity, 
                    Compositions.NucID
             FROM Resources 
             INNER JOIN Transactions ON Transactions.ResourceID = Resources.ResourceID 
	     INNER JOIN Compositions ON Compositions.StateID = Resources.ResourceID
             WHERE Transactions.ReceiverID=23 
             ORDER BY Resources.TimeCreated;"""

    cur = c.execute(sql)
    results = cur.fetchall()

    return results


