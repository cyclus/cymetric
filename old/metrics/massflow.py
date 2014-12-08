import sqlite3

def fuelmass(c):
    """Lists mass (in kg) of enriched fuel transacted from with respect to nuclide and time.

    Args:
        c: connection cursor to sqlite database.
    """
    sql = """SELECT resources.TimeCreated, compositions.MassFrac*resources.Quantity, 
                    compositions.NucId
             FROM resources 
             INNER JOIN transactions ON transactions.ResourceId =
resources.ResourceId 
	     INNER JOIN compositions ON compositions.QualId =
resources.ResourceId
             WHERE transactions.ReceiverId=13 
             ORDER BY resources.TimeCreated;"""

    cur = c.execute(sql)
    results = cur.fetchall()

    return results


