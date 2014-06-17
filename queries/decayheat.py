import csv
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
    sql = ("SELECT resources.TimeCreated, compositions.NucId," 
           "compositions.MassFrac*resources.Quantity ")
    sql += ("FROM resources "
            "INNER JOIN compositions ON resources.QualId = compositions.QualId "
            "INNER JOIN transactions ON resources.TimeCreated = transactions.Time "
            "WHERE transactions.SenderId=13 "
            "GROUP BY resources.TimeCreated, compositions.NucId "
            "ORDER BY resources.TimeCreated;")
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
        exp = t * data.decay_const(nuc)
        q_i = dh * 0.5**exp
        end_heat += q_i

    return end_heat    

def decayheat(c):
    """Lists decay heats at 0, 10, 100, 1000, 10000 years summed over all
       nuclides and facilities.

    Args:
        c: connection cursor to sqlite database.
    """
    # Conversion of time from months to seconds
    MCONV = 3.16e7 / 12

    # Conversion of years to seconds
    YCONV = 3.16e7

    # Retrieve list of decay heats of each nuclide
    alldecayheats = query(c)

    dict_decayheat = {}
    sim_time = alldecayheats[-1][0]
    # Get only one nuclide per entry by adding decay heats
    for time_step, nuc, dh in alldecayheats:
        sec = (sim_time - time_step) * MCONV
        q_i = dh * 0.5**(sec * data.decay_const(nuc))
        if nuc in dict_decayheat.keys():
            dict_decayheat[nuc] += q_i
        else:
            dict_decayheat[nuc] = q_i

    # Put back into list of tuples & sort by time-step
    decayheat = dict_decayheat.items()
    decayheat.sort()

    # calculate decay heats of each nuc 0, 10, 100, 1000, 10000 yrs after sim
    decayheats = []
    for nuc, dh0 in decayheat:
        sec10 = 10 * YCONV
        sec100 = 100 * YCONV
        sec1000 = 1000 * YCONV
        sec10000 = 10000 * YCONV
        dh10 = dh0 * 0.5**(sec10 * data.decay_const(nuc))
        dh100 = dh0 * 0.5**(sec100 * data.decay_const(nuc))
        dh1000 = dh0 * 0.5**(sec1000 * data.decay_const(nuc))
        dh10000 = dh0 * 0.5**(sec10000 * data.decay_const(nuc))
        row = (nuc, dh0, dh10, dh100, dh1000, dh10000)
        decayheats.append(row)

    # Write to csv file 
    fname = 'decayheat.csv'
    with open(fname,'w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['nuclide',
                          'decay heat at 0 yrs [MW]',
                          'decay heat at 10 yrs [MW]',
                          'decay heat at 100 yrs [MW]',
                          'decay heat at 1000 yrs [MW]',
                          'decay heat at 10000 yrs [MW]'])
        for row in decayheats:
            csv_out.writerow(row)

    print('file saved as ' + fname + '!')
