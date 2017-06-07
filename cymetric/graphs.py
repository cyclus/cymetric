import warnings

import pandas as pd
import numpy as np

try:
    from graphviz import Digraph
    HAVE_GRAPHVIZ = True
except ImportError:
    HAVE_GRAPHVIZ = False

try:
    from pyne import data
    import pyne.enrichment as enr
    from pyne import nucname
    HAVE_PYNE = True
except ImportError:
    HAVE_PYNE = False


import cymetric as cym
from cymetric import tools
from cymetric.filter import get_transaction_nuc_df

def get_flow_graph(evaler, senders=(), receivers=(), commodities=(), nucs=(),
                   time=[-1, -1]):
    """
    Generate the dot graph of the transation between facilitiese. Applying times
    nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    senders :  of the sending facility
    receivers :  of the receiving facility
    commodities :  of the commodity exchanged
    nucs :  of nuclide to select.
    """
    tools.raise_no_graphviz('Unable to generate flow graph!', HAVE_GRAPHVIZ)

    df = get_transaction_nuc_df(
        evaler, senders, receivers, commodities, nucs)

    if time[0] != -1:
        df = df.loc[(df['Time'] > time[0])]
    if time[1] != -1:
        df = df.loc[(df['Time'] < time[1])]

    group_end = ['ReceiverPrototype', 'SenderPrototype']
    group_start = group_end + ['Mass']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    agents_ = evaler.eval('AgentEntry')['Prototype'].tolist()

    dot = Digraph('G')

    for agent in agents_:
        dot.node(agent)

    for index, row in df.iterrows():
        dot.edge(row['SenderPrototype'], row['ReceiverPrototype'],
                 label=str(row['Mass']))

    return dot
