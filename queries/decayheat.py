import sqlite3
from pyne import data

def query(c):
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

    return alldecayheats

def enddecayheat(c):
    """Lists decay heat at end of simulation.

    Args:
        c: connection cursor to sqlite database.
    """
    # Conversion of time from months to seconds
    CONV = 3.16e7 / 12

    # Retrieve list of decay heats of each nuclide
    alldecayheats = query(c)

    end_heat = 0
    sim_time = alldecayheats[-1][0]

    # Sum decayed heats for each time-step/nuclide
    for time_step, nuc, dh in alldecayheats:
        t = (sim_time - time_step) * CONV
        exp = t / data.decay_const(nuc)
        q_i = dh * 0.5**exp
        end_heat += q_i

    return end_heat    

def decayheat(c):
    """Lists decay heats at each time-step for all facilities.

    Args:
        c: connection cursor to sqlite database.
    """
    # Retrieve list of decay heats of each nuclide
    alldecayheats = query(c)

    dict_decayheats = {}
    decayheats = []

    # Get only one time-step per entry, add decay heats of same time-steps
    for time_step, nuc, dh in alldecayheats:
        if time_step in dict_decayheats.keys():
            dict_decayheats[time_step] += dh
        else:
            dict_decayheats[time_step] = dh

    # Put back into list of tuples & sort by time-step
    decayheats = dict_decayheats.items()
    decayheats.sort()

    return decayheats
