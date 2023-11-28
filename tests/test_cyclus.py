"""Tests for cyclus wrappers"""

from tools import dbtest


@dbtest
def test_name(db, fname, backend):
    obs = db.name
    assert fname == obs


@dbtest
def test_simid(db, fname, backend):
    df = db.query("AgentEntry")
    simid = df['SimId']
    exp = simid[0]
    for obs in simid:
        assert exp == obs


@dbtest
def test_conds_ae(db, fname, backend):
    obs = db.query("AgentEntry", [('Kind', '==', 'Region')])
    assert 1 == len(obs)
    assert 'Region' == obs['Kind'][0]
    assert ':agents:NullRegion' == obs['Spec'][0]


@dbtest
def test_conds_comp(db, fname, backend):
    conds = [('NucId', '==', 922350000), ('MassFrac', '>', 0.0072)]
    df = db.query("Compositions", conds)
    assert 0 < len(df)
    for row in df['MassFrac']:
        assert 0.0072 < row

