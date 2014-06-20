import csv
import sqlite3
import math
from pyne import data

def query(c):
    """Lists activities of all nuclides with respect to time for all facilities.

    Args:
        c: connection cursor to sqlite database.
    """

    # SQL query returns a table with the nuclides and their masses at each timestep
    sql = ("SELECT resources.TimeCreated, compositions.NucID," 
           "compositions.MassFrac*resources.Quantity ")
    sql += ("FROM resources "
            "INNER JOIN compositions ON resources.QualId = compositions.QualId "
            "GROUP BY resources.TimeCreated, compositions.NucId "
            "ORDER BY resources.TimeCreated;")
    cur = c.execute(sql)
    results = cur.fetchall()

    # Gives avogadro's number with a kg to g conversion 
    CONV = 1000*6.022e23

    activities = []

    # Calculates activities (/s) of each nuclide at each timestep
    for time_step, nuc, mass in results:
        act = CONV * mass * data.decay_const(nuc) / data.atomic_mass(nuc)
        row = (time_step, nuc, mass, act)
        activities.append(row)

    return activities

def activity(c):
    """Lists activities of all nuclides at 10, 100, 1000, 10000 yrs from the end of the sim.

    Args:
        c: connection cursor to sqlite database.
    """

    activities = query(c)

    # Conversion of time from months to seconds
    MCONV = 3.16e7 / 12

    # Conversion of years to seconds
    YCONV = 3.16e7

    dict_acts = {}
    sim_time = activities[-1][0]
    # Get only one nuclide per entry, add activities @ end of sim
    for time_step, nuc, mass, act in activities:
        sec = (sim_time - time_step) * MCONV
        acts = act * math.exp(-sec * data.decay_const(nuc))
        if nuc in dict_acts.keys():
            dict_acts[nuc] += acts
        else:
            dict_acts[nuc] = acts

    # Put back into list of tuples & sort by nuclide
    act_endsim = dict_acts.items()
    act_endsim.sort()

    # calculate activities 10, 100, 1000, 10000 yrs later
    acts = [] 
    for nuc, act0 in act_endsim:
        sec10 = 10 * YCONV
        sec100 = 100 * YCONV
        sec1000 = 1000 * YCONV
        sec10000 = 10000 * YCONV
        act10 = act0 * math.exp(-sec10 * data.decay_const(nuc))
        act100 = act0 * math.exp(-sec100 * data.decay_const(nuc))
        act1000 = act0 * math.exp(-sec1000 * data.decay_const(nuc))
        act10000 = act0 * math.exp(-sec10000 * data.decay_const(nuc))
        row = (nuc, act0, act10, act100, act1000, act10000)
        acts.append(row)

    # Write to csv file 
    fname = 'activity.csv'
    with open(fname,'w') as out:
        csv_out=csv.writer(out)
	csv_out.writerow(['nuclide', 
                          'act at 0 yrs [Bq]', 
                          'act at 10 yrs [Bq]', 
                          'act at 100 yrs [Bq]', 
                          'act at 1000 yrs [Bq]', 
                          'act at 10000 yrs [Bq]'])
        for row in acts:
            csv_out.writerow(row)

    print('file saved as ' + fname + '!')
