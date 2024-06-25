"""A plot generator for Cymetric.
"""
import warnings


try:
    from graphviz import Digraph
    HAVE_GRAPHVIZ = True
except ImportError:
    HAVE_GRAPHVIZ = False

from cymetric import tools
from cymetric.filters import transactions_nuc


def flow_graph(evaler, senders=(), receivers=(), commodities=(), nucs=(),
               label='', start=None, stop=None):
    """
    Generate the dot graph of the transation between facilitiese. Applying times
    nuclides selection when required.

    Parameters
    ----------
    evaler : evaler
    senders : list of the sending facility to consider
    receivers : list of the receiving facility to consider
    commodities : list of the commodity exchanged to consider
    nucs : list of nuclide to consider
    label : label key, used to add label on the arrow connecting facilities (for
    commodity use 'com', for mass use 'mass', for both use com,mass)
    start : first timestep to consider, start included
    stop : last timestep to consider, stop included
    """
    tools.raise_no_graphviz('Unable to generate flow graph!', HAVE_GRAPHVIZ)

    df = transactions_nuc(
        evaler, senders, receivers, commodities, nucs)

    if start is not None:
        df = df.loc[(df['Time'] >= start)]
    if stop is not None:
        df = df.loc[(df['Time'] <= stop)]

    group_end = ['ReceiverPrototype', 'SenderPrototype', 'Commodity']
    group_start = group_end + ['Mass']
    df = df[group_start].groupby(group_end).sum()
    df.reset_index(inplace=True)

    agents_ = evaler.eval("AgentEntry")

    dot = Digraph("G", strict=True)

    # start by constructing region subgraphs
    regions = agents_[agents_["ParentId"] == -1]
    for i, row in regions.iterrows():
        region_id = row["AgentId"]
        region_prototype = row["Prototype"]
        institutions = agents_[agents_["ParentId"] == region_id]
        # graphviz requires subgraphs start with the prefix cluster_
        with dot.subgraph(name=f"cluster_{region_id}") as c:
            c.attr(
                style="dotted",
                label=region_prototype,
                color="black",
            )
            # then construct institution subgraphs
            for j, institution in institutions.iterrows():
                institution_id = institution["AgentId"]
                institution_prototype = institution["Prototype"]
                # graphviz requires subgraphs start with the prefix cluster_
                with c.subgraph(name=f"cluster_{institution_id}") as b:
                    b.attr(
                        style="filled",
                        label=institution_prototype,
                        color="lightgray",
                    )
                    # facilities are nodes in the (sub)graph(s)
                    facilities = agents_[agents_["ParentId"] == institution_id]
                    for k, facility in facilities.iterrows():
                        facility_id = facility["AgentId"]
                        facility_prototype = facility["Prototype"]
                        b.node(
                            name=str(facility_id),
                            label=str(facility_prototype),
                        )

    # use transactions to determine edges
    for index, row in df.iterrows():
        lbl = ""
        if "com" in label:
            lbl += str(row["Commodity"]) + " "
        if "mass" in label:
            lbl += str("{:.2e}".format(row["Mass"])) + " "
        dot.edge(
            str(row["SenderId"]),
            str(row["ReceiverId"]),
            label=lbl,
        )

    return dot

