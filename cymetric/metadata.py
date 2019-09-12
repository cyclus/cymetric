import warnings
import pandas as pd
import numpy as np


try:
    from pyne import data
    import pyne.enrichment as enr
    from pyne import nucname
    HAVE_PYNE = True
except ImportError:
    HAVE_PYNE = False


from cymetric.tools import format_nucs, reduce, merge, add_missing_time_step

def get_usage(evaler):

    metadata = evaler.eval("Metadata")
    agents_entry = evaler.eval('AgentEntry')
    deployment = metadata[ metadata['Type'] == "deployment" ]
    decom = metadata[ metadata['Type'] == "decommission" ]
    timestep = metadata[ metadata['Type'] == "timestep" ]
    throughput_meta = metadata[ metadata['Type'] == "throughput" ]

    worklabel = metadata[ metadata['Keyword'] == "WORKLABEL" ]

    # Deployement
    dep_agent = agents_entry[agents_entry['AgentId'].isin(deployment['AgentId'])]
    _tmp = pd.merge(deployment[['SimId', 'AgentId','Keyword', 'Value']], dep_agent, on=['SimId', 'AgentId'])


    deployment_use = pd.DataFrame(data={'SimId': _tmp.SimId,
                                        'AgentId': _tmp.AgentId,
                                        'Time': _tmp.EnterTime,
                                        'Keyword': _tmp.Keyword,
                                        'Value':_tmp.Value.astype(float)},
                                  columns=['SimId', 'AgentId', 'Time', 'Keyword', 'Value'])
    rtn = deployment_use.copy()

    # Decommision
    decom_agent = agents_entry[agents_entry['AgentId'].isin(decom['AgentId'])]
    decom_agent = decom_agent.reset_index(drop=True)
    decom_agent['ExitTime'] = decom_agent['EnterTime'] + decom_agent['Lifetime']
    _tmp = pd.merge(decom[['SimId', 'AgentId','Keyword', 'Value']], decom_agent, on=['SimId', 'AgentId'])
    decom_use = pd.DataFrame(data={'SimId': _tmp.SimId,
                                   'AgentId': _tmp.AgentId,
                                   'Time': _tmp.ExitTime,
                                   'Keyword': _tmp.Keyword,
                                   'Value':_tmp.Value.astype(float)},
                             columns=['SimId', 'AgentId', 'Time', 'Keyword', 'Value'])
    rtn = pd.concat([rtn, decom_use], ignore_index=True)

    # TimeStep
    timestep_agent = agents_entry[agents_entry['AgentId'].isin(timestep['AgentId'])]
    timestep_agent = timestep_agent.reset_index(drop=True)
    timestep_agent['ExitTime'] = timestep_agent['EnterTime'] + timestep_agent['Lifetime']
    timestep_tmp = pd.DataFrame(data={'SimId': _tmp.SimId,
                                      'AgentId': _tmp.AgentId,
                                      'EnterTime': _tmp.EnterTime,
                                      'ExitTime': _tmp.ExitTime ,
                                      'Keyword': _tmp.Keyword,
                                      'Value':_tmp.Value.astype(float)},
                              columns=['SimId', 'AgentId', 'EnterTime', 'ExitTime', 'Keyword', 'Value'])
    time_step_data = []
    for index, row  in timestep_tmp.iterrows():
        for i in range(row['EnterTime'], row['ExitTime']):
            time_step_data.append( (row['SimId'],
                          row['AgentId'],
                          i,
                          row['Keyword'],
                          row['Value']))
    timestep_use = pd.DataFrame(time_step_data, columns=['SimId', 'AgentId', 'Time', 'Keyword', 'Value'])
    rtn = pd.concat([rtn, timestep_use], ignore_index=True)

    worklabel.drop(columns=['AgentId'], inplace=True)
    worklabel.drop_duplicates(inplace=True)
    worktimeseries = []
    for index, row in worklabel.iterrows():
        if (row['AgentId'] in throughput_meta['AgentId']):
            work_name = "TimeSeries" + row['Value']
            timeseries = myEval.eval(work_name)
            if timeseries is not None:
                worktimeseries.append(timeseries[timeseries['AgentId'] == row['AgentId'] ])

    def get_throughput_timeseries(throughput_df, throughput_meta):
        if throughput_df is not None:
            _tmp = pd.merge(throughput_meta[['SimId', 'AgentId','Keyword', 'Value']], throughput_df, on=['SimId', 'AgentId'])
            _tmp['Value_y'] = _tmp.Value_x.astype(float)*_tmp.Value_y
            _tmp.drop(columns=['Value_x'], inplace=True)
            _tmp.rename(columns={"Value_y": "Value"}, inplace=True)
            return _tmp
        else:
            return pd.DataFrame()

    for work in worktimeseries:
        rtn = pd.concat([rtn, get_throughput_timeseries(work, throughput_meta)], ignore_index=True)


    return rtn
