"""Tests for cyclus wrappers"""

from tools import dbtest



def test_name(dbtest):
    db, fname, backend = dbtest
    obs = db.name
    assert fname == obs



def test_simid(dbtest):
    db, fname, backend = dbtest
    df = db.query("AgentEntry")
    simid = df['SimId']
    exp = simid[0]
    for obs in simid:
        assert exp == obs


def test_conds_ae(dbtest):
    db, fname, backend = dbtest
    obs = db.query("AgentEntry", [('Kind', '==', 'Region')])
    assert 1 == len(obs)
    assert 'Region' == obs['Kind'][0]
    assert ':agents:NullRegion' == obs['Spec'][0]


def test_conds_comp(dbtest):
    db, fname, backend = dbtest
    conds = [('NucId', '==', 922350000), ('MassFrac', '>', 0.0072)]
    df = db.query("Compositions", conds)
    assert 0 < len(df)
    for row in df['MassFrac']:
        assert 0.0072 < row

