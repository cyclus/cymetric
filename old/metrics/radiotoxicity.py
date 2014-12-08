import csv
import sqlite3
import math
from pyne import data

import activity

def rel_activity(c):
    """Lists activity of spent fuel from all facilities relative to natural U 
       0, 10, 100, 1000, 10000 years after the end of the simulation.

    Args:
        c: connection cursor to sqlite database.
    """

    activities = activity.query(c)
    dict_acts = {}
    sim_time = activities[-1][0]

    # Conversion of time from months to seconds
    MCONV = 3.16e7 / 12

    # Conversion of years to seconds
    YCONV = 3.16e7

    dict_acts = {}
    tot_mass = 0.0
    sim_time = activities[-1][0]
    # Get list of one activity per nuclide wrt end of sim & sum total mass
    for time_step, nuc, mass, act in activities:
        tot_mass += mass
        sec = (sim_time - time_step) * MCONV
        acts = act * math.exp(-sec * data.decay_const(nuc))
        if nuc in dict_acts.keys():
            dict_acts[nuc] += acts
        else:
            dict_acts[nuc] = acts

    # Put back into list of tuples & sort by nuclide
    act_endsim = dict_acts.items()
    act_endsim.sort()

    # calculate natural uranium activity
    CONV_235 = 0.007*1000*6.022e23
    CONV_238 = 0.993*1000*6.022e23
    actU235 = CONV_235 * tot_mass * data.decay_const('U235') / data.atomic_mass('U235')
    actU238 = CONV_238 * tot_mass * data.decay_const('U235') / data.atomic_mass('U235')
    act_U = actU235 + actU238

    # calculate relative activities to nat U after 0, 10, 100, 1000, 10000 yrs
    rel_acts = []
    for nuc, act0 in act_endsim:
        t = 10
        nuc_acts = (act0,)
        while t <= 10000:
            sec = t * YCONV
            nuc_act = act0 * math.exp(-sec * data.decay_const(nuc))
            nuc_acts += (nuc_act,)
            t = 10 * t
        rel = []
        for i in nuc_acts:
            frac = i / act_U
            rel.append(frac)
        row = (nuc,) + tuple(rel)
        rel_acts.append(row)

    # Write to csv file 
    fname = 'relative_activity.csv'
    with open(fname,'w') as out:
        csv_out=csv.writer(out)
	csv_out.writerow(['nuclide', 
                          'rel_act at 0 yrs', 
                          'rel_act at 10 yrs', 
                          'rel_act at 100 yrs', 
                          'rel_act at 1000 yrs', 
                          'rel_act at 10000 yrs'])
        for row in rel_acts:
            csv_out.writerow(row)

    print('file saved as ' + fname + '!')


def radiotoxicity(c):
    """Lists radiotoxicity for all facilities 0, 10, 100, 1000, 10000 years
       after the end of the simulation.

    Args:
        c: connection cursor to sqlite database.
    """

    # fetches list of activities for all nuclides at each time step
    activities = activity(c)

    for nuc, act0, act10, act100, act1000, act10000 in activities:
        nuc = 0

