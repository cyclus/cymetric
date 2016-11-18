from pyne import nucname
import pandas as pd
import numpy as np
import cymetric as cym

def GetTransactionTimeSeries(db, sender='All', receiver='All', *args):
    
    nuc_list = []
    for inx, nuc in enumerate(args):
        nuc_list.append(nucname.id(nuc))

    # initiate evaluation
    evaler = cym.Evaluator(db)
    
    # get transation & Agent tables
    trans = evaler.eval('Transactions')
    agents = evaler.eval('AgentEntry')
    
# build 2 table for SenderId and ReceiverId
    # get receiver
    agents_receiver = agents.rename(index=str, columns={'AgentId': 'ReceiverId'})
    if receiver != 'All':
        agents_receiver_ = agents_receiver.loc[lambda df: df.Prototype == receiver,:]
        # check if receiver exists
        if agents_receiver_.empty:
            print("unknown Receiver, available Receiver are:")
            for receiver_name in agents_receiver.Prototype.unique(): # check if the loop is correct should it not be trans.XX.XX
                print(receiver_name)
        else:
            agents_receiver = agents_receiver_
    # get sender
    agents_sender = agents.rename(index=str, columns={'AgentId': 'SenderId'})
    if sender != 'All':
        agents_sender_ = agents_sender.loc[lambda df: df.Prototype == sender,:]
        # check if sender exists
        if agents_sender_.empty:
            print("unknown Sender, available Sender are:")
            for sender_name in agents_sender.Prototype.unique():
                print(sender_name)
        else:
            agents_sender = agents_sender_

    # check if sender and receiver exist
    if( agents_sender.empty or agents_receiver.empty):
        return 0
    else: 
    # Both receiver and sender exist:
        # select good corresponding transaction
        selected_trans = trans.loc[trans['ReceiverId'].isin(agents_receiver.ReceiverId)]
        selected_trans = selected_trans.loc[selected_trans['SenderId'].isin(agents_sender.SenderId)]
        
        # Merge Sender infos
        df = pd.merge(agents_sender[['SimId', 'SenderId', 'Prototype']], selected_trans, on=['SimId', 'SenderId'])
        df = df.rename(index=str, columns={'Prototype': 'SenderProto'})
        df = df.drop('SenderId',1)
        
        # Merge reveiver infos
        df = pd.merge(agents_receiver[['SimId', 'ReceiverId', 'Prototype']], df, on=['SimId', 'ReceiverId'])
        df = df.rename(index=str, columns={'Prototype': 'ReceiverProto'})
        df = df.drop('ReceiverId',1)

        # Get resource and select the proper one
        resource = evaler.eval('Resources')
        selected_resources = resource.loc[resource['ResourceId'].isin(selected_trans.ResourceId)]

        # merge Resource into transaction
        df = pd.merge(selected_resources[['SimId', 'ResourceId','QualId','Quantity','Units'    ]], df, on=['SimId', 'ResourceId'])
        df = df.drop('ResourceId',1)

        if len(nuc_list) != 0: # Nuclide required -> do the Math
            compo = evaler.eval('Compositions')
            selected_compo = compo.loc[compo['NucId'].isin(nuc_list)]

            df = pd.merge(selected_compo[['SimId', 'QualId','NucId','MassFrac'    ]], df, on=['SimId', 'QualId'])
            df['Quantity'] = df['Quantity'] * df['MassFrac']
        
        grouped_trans = df[['ReceiverProto', 'SenderProto','Time', 'Quantity']].groupby(['ReceiverProto', 'SenderProto','Time']).sum()

        if sender == 'All':
            grouped_trans = df[['ReceiverProto','Time', 'Quantity']].groupby(['ReceiverProto','Time']).sum()
            trans_table = grouped_trans.loc[receiver]
        elif receiver == 'All':
            grouped_trans = df[['SenderProto','Time', 'Quantity']].groupby(['SenderProto','Time']).sum()
            trans_table = grouped_trans.loc[sender]
        else:
            trans_table = grouped_trans.loc[receiver].loc[sender]
    
    return trans_table


def GetInventoryTimeSeries(db, facility, *args):
    
    nuc_list = []
    for inx, nuc in enumerate(args):
        nuc_list.append(nucname.id(nuc))
    
    #initiate evaluation
    evaler = cym.Evaluator(db)
    
    # Get inventory table
    inv = evaler.eval('ExplicitInventory')
    if len(nuc_list) != 0 :
        inv = inv.loc[inv['NucId'].isin(nuc_list)]
    agents = evaler.eval('AgentEntry')

    selected_agents = agents.loc[lambda df: df.Prototype == facility,:]
    if selected_agents.empty:
        print("unknown Facitlity, available Facilities are:")
        for fac_name in agents.Prototype.unique():
            print(fac_name)
        inv_table = 0
    else:
        selected_inv = inv.loc[inv['AgentId'].isin(selected_agents.AgentId)]
        
        df = pd.merge(selected_agents[['SimId', 'AgentId', 'Prototype']], selected_inv, on=['SimId', 'AgentId'])
        df = df.drop('AgentId',1)
        
        inv_table = df[['Prototype', 'Time','Quantity']].groupby(['Prototype', 'Time']).sum()

    return inv_table
