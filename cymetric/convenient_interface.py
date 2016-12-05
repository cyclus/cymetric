from pyne import nucname
import pandas as pd
import numpy as np
import cymetric as cym
import warnings



def select_agent(agent_name, agent, agent_type = "Agent"):
    if agent_name != 'All':
        agents_selected = agents.loc[
            lambda df: df.Prototype == agent_name, :]
        # check if exists
        if agent_selected.empty:
            wng_msg = "unknown " + agent_type + ", available Agents are:"
            for receiver_name in agents.Prototype.unique():
                wng_msg += " " + receiver_name
            warnings.wrn(wng_msg, UserWarning)
        else:
            agents_pdf = agents_selected
    return agents

def merge_n_drop(pdf, base_col, add_pdf, add_col):
    pdf = pd.merge(add_pdf[add_col], pdf, on = base_col);
    pdf.drop(base_col[1],1)
    return pdf


def get_transaction_timeseries(db, send_name='All', rec_name='All'):


    # initiate evaluation
    evaler = cym.Evaluator(db)

    # get transation & Agent tables
    trans = evaler.eval('Transactions')
    agents = evaler.eval('AgentEntry')
    resources=evaler.eval('Resources')

    # build 2 table for SenderId and ReceiverId
    # get receiver
    rec_list = agents.rename(index=str, columns={'AgentId': 'ReceiverId'})
    rec_list = select_agent(receiver, rec_list, "Receiver")

    # get sender
    send_list = agents.rename(index=str, columns={'AgentId': 'SenderId'})
    send_list = select_agent(send_name, send_list, "Sender")
    
    # check if sender and receiver exist
    if rec_list.empty or send_list.empty:
        return None
    else:
    # Both receiver and sender exist:
        trans = trans.loc[trans['ReceiverId'].isin(rec_list.ReceiveriId)]
        trans = trans.loc[trans['SenderId'].isin(send_list.SenderId)]
        resources=resources.loc[resource['ResourceId'].isin(trans.ResourceId)]

        df = merge_n_drop(trans, ['SimId', 'SenderId'], send_list, ['SimId', 'SenderId', 'Prototype'])
        df= df.rename(index = str, columns = {'Prototype': 'SenderProto'}) 

        df = merge_n_drop(trans, ['SimId', 'ReceiverId'], rec_list, ['SimId', 'ReceiverId', 'Prototype'])
        df= df.rename(index = str, columns = {'Prototype': 'ReceiverProto'}) 

        df = merge_n_drop(trans, ['SimId', 'RessourceId'], resources, ['SimId', 'ResourceId', 'QualId', 'Quantity', 'Unit'])
    
    return df

def get_transaction_timeseries(db, send_name='All', rec_name='All', *args):

    df = get_transaction_timeseries(db, send_name, rec_name)
        group_trans=df[['ReceiverProto', 'SenderProto', 'Time', 'Quantity']].groupby(
            ['ReceiverProto', 'SenderProto', 'Time']).sum()

        if sender == 'All':
            grouped_trans=df[['ReceiverProto', 'Time', 'Quantity']].groupby(
                ['ReceiverProto', 'Time']).sum()
            trans_table=grouped_trans.loc[receiver]
        elif receiver == 'All':
            grouped_trans=df[['SenderProto', 'Time', 'Quantity']].groupby(
                ['SenderProto', 'Time']).sum()
            trans_table=grouped_trans.loc[sender]
        else:
            trans_table=grouped_trans.loc[receiver].loc[sender]

    return trans_table


def GetInventoryTimeSeries(db, facility, *args):

    nuc_list = []
    for inx, nuc in enumerate(args):
        nuc_list.append(nucname.id(nuc))

    # initiate evaluation
    evaler = cym.Evaluator(db)

    # Get inventory table
    inv = evaler.eval('ExplicitInventory')
    if len(nuc_list) != 0:
        inv = inv.loc[inv['NucId'].isin(nuc_list)]
    agents = evaler.eval('AgentEntry')

    selected_agents = agents.loc[lambda df: df.Prototype == facility, :]
    if selected_agents.empty:
        print("unknown Facitlity, available Facilities are:")
        for fac_name in agents.Prototype.unique():
            print(fac_name)
        inv_table = None
    else:
        selected_inv = inv.loc[inv['AgentId'].isin(selected_agents.AgentId)]

        df = pd.merge(selected_agents[
                      ['SimId', 'AgentId', 'Prototype']], selected_inv, on=['SimId', 'AgentId'])
        df = df.drop('AgentId', 1)

        inv_table = df[['Prototype', 'Time', 'Quantity']
                       ].groupby(['Prototype', 'Time']).sum()

    return inv_table
