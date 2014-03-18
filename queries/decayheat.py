import sqlite3
from pyne import data

def decayheat(c):
    """Lists decay heats of all nuclides with respect to time for all facilities.

    Args:
        c: connection cursor to sqlite database.
    """
    
    ACT_CONV = 1000*6.022e23
    Q_CONV = 1.602e-19
    
    # SQL query returns a table with the nuclides (and their masses) transacted from reactor
    sql = ("SELECT Resources.TimeCreated, Compositions.NucID," 
           "Compositions.MassFrac*Resources.Quantity ")
    sql += ("FROM Resources "
            "INNER JOIN Compositions ON Resources.StateID = Compositions.StateID "
            "INNER JOIN Transactions ON Resources.TimeCreated = Transactions.Time "
            "WHERE Transactions.SenderID=23 "
            "GROUP BY Resources.TimeCreated, Compositions.NucID "
            "ORDER BY Resources.TimeCreated;")
    cur = c.execute(sql)
    results = cur.fetchall()

    decayheats = []
    prev_line = []
    
    # Calculates decay heat (MW) at each timestep
    for time_step, nuc, mass in results:
        act = ACT_CONV * mass * data.decay_const(nuc) / data.atomic_mass(nuc)
        dh = Q_CONV * act * data.q_val(nuc)
        curr_line = [time_step, nuc, mass]

        # Sums decay heats for each time step
        if curr_line[0] == prev_line[0]:
            dh += prev_line[2]
        row = (time_step, dh)
        decayheats.append(row)
        prev_line = curr_line
        
    return decayheats
