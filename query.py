import sqlite3

def all_agents(c, sim_id):
    """List of all agents and their info.

    Args:
        c: connection cursor to sqlite database.
        sim_id: simulation ID
    """
    sql = """SELECT ID,AgentType,ModelType,Prototype,ParentID,EnterDate,DeathDate FROM
                Agents INNER JOIN AgentDeaths ON Agents.ID = AgentDeaths.AgentID
                WHERE Agents.SimID = """ + sim_id + " AND Agents.SimID = AgentDeaths.SimID;"
    return c.execute(sql)

def deploy_cumulative(c, sim_id, prototype_name):
    """Time-series of # active deployments of a specific prototype.

    Args:
        c: connection cursor to sqlite database.
        sim_id: simulation ID.
        prototype_name: name of a prototype.
    """
    sql = """SELECT ti.Time,COUNT(*)
              FROM Agents AS ag
              INNER JOIN AgentDeaths AS ad ON ag.ID = ad.AgentID
              INNER JOIN TimeList AS ti ON ti.Time >= ag.EnterDate AND ad.DeathDate > ti.Time
            WHERE
              ag.SimID = """ + sim_id + """ AND ag.SimID = ad.SimID
              AND ag.Prototype = """ + prototype_name + """
            GROUP BY ti.Time
            ORDER BY ti.Time;"""
    return c.execute(sql)

def inv_series(c, sim_id, agent_id, iso_id):
    """Timeseries of a specific agent's inventory of a specific isotope.

    Args:
        c: connection cursor to sqlite database.
        sim_id: simulation ID.
        agent_id: ID of an agent.
        iso_id: ID of an isotope/nuclide.
    """
    sql = """SELECT ti.Time,SUM(cmp.Quantity * inv.Quantity) FROM (
                Compositions AS cmp
                INNER JOIN Inventories AS inv ON inv.StateID = cmp.ID
                INNER JOIN TimeList AS ti ON (ti.Time >= inv.StartTime AND ti.Time < inv.EndTime)
            ) WHERE (
                 inv.SimID = """ + sim_id + """ AND inv.SimID = cmp.SimID
                 AND inv.AgentID = """ + agent_id + "AND cmp.IsoID = " + iso_id +"""
            ) GROUP BY ti.Time,cmp.IsoID;"""
    return c.execute(sql)

def inv_at(c, sim_id, agent_id, t1, t2):
    """Total inventory(all isotopes) of a specific agent at specific timestep.

    Args:
        c: connection to sqlite database.
        sim_id: simulation ID.
        agent_id: ID of an agent.
        t1: start time.
        t2: end time.
    """
    sql = """SELECT cmp.IsoID,SUM(cmp.Quantity * inv.Quantity) FROM (
                Inventories AS inv
                INNER JOIN Compositions AS cmp ON inv.StateID = cmp.ID
            ) WHERE (
                inv.SimID = """ + sim_id + """ AND inv.SimID = cmp.SimID
                AND inv.StartTime <= """ + t1 + " AND inv.EndTime > " + t2 + """
                AND inv.AgentID = """ + agent_id + """
            ) GROUP BY cmp.IsoID;"""
    return c.execute(sql)

def mat_created(c, sim_id, agent_id, t1, t2):
    """Total amount of material(all isotopes) created by a particular agent
    between two timesteps.

    Args:
        c: connection to a sqlite database.
        sim_id: simulation ID.
        agent_id: ID of an agent.
        t1: start time.
        t2: end time.
    """
    sql = """SELECT cmp.IsoID,SUM(cmp.Quantity * res.Quantity) FROM (
                Resources As res
                INNER JOIN Compositions AS cmp ON res.StateID = cmp.ID
                INNER JOIN ResCreators AS cre ON res.ID = cre.ResID
            ) WHERE (
                cre.SimID = """ + sim_id + """ AND cre.SimID = res.SimID AND cre.SimID = cmp.SimID
                AND res.TimeCreated >= """ + t1 + " AND res.TimeCreated < " + t2 + """
                AND cre.ModelID = """ + agent_id + """
            ) GROUP BY cmp.IsoID;"""
    return c.execute(sql)

def flow(c, sim_id, from_agent_id, to_agent_id, t1, t2):
    """Total material(all isotopes) transacted between two agents between two
    timesteps.

    Args:
        c: connection to sqlite database.
        sim_id: simulation ID.
        from_agent_id: ID of a sending agent.
        to_agent_id: ID of a receiving agent.
        t1: start time.
        t2: end time.
    """
    sql = """SELECT cmp.IsoID,SUM(cmp.Quantity * res.Quantity) FROM (
               Resources AS res
               INNER JOIN Compositions AS cmp ON cmp.ID = res.StateID
               INNER JOIN Transactions AS tr ON tr.ResourceID = res.ID
             ) WHERE (
               res.SimID = """ + sim_id + """ AND cmp.SimID = res.SimID AND tr.SimID = res.SimID
               AND tr.Time >= """ + t1 + " AND tr.Time < " + t2 + """
               AND tr.SenderID = """ + from_agent_id + " AND tr.ReceiverID = " + to_agent_id + """
             ) GROUP BY cmp.IsoID;"""
    return c.execute(sql)

