import sqlite3
from pyne import data

def decayheat(c):
    """Lists decay heats of all nuclides with respect to time for all facilities.

    Args:
        c: connection cursor to sqlite database.
    """
    # gives avogadro's number with a kg to g conversion
    ACT_CONV = 1000*6.022e23
    # converts from MeV/s to MW
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

    alldecayheats = []

    # Calculates decay heat (MW) at each timestep
    for time_step, nuc, mass in results:
        act = ACT_CONV * mass * data.decay_const(nuc) / data.atomic_mass(nuc)
        dh = Q_CONV * act * data.q_val(nuc)
        row = (time_step, nuc, dh)
        alldecayheats.append(row)
        
    # Sums decay heats for each time step
    dict_decayheats = {}
    decayheats = []

    for time_step, nuc, dh in alldecayheats:
        if time_step in dict_decayheats.keys():
            dict_decayheats[time_step] += dh
        else:
            dict_decayheats[time_step] = dh
    
    decayheats = dict_decayheats.items()
    decayheats.sort()

    return decayheats
