"""Tests for root metrics"""
from __future__ import print_function, unicode_literals

from cymetric import root_metrics

from tools import dbtest


@dbtest
def test_resources(db, fname, backend):
    r = root_metrics.resources(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'Resources' == r.name


@dbtest
def test_compositions(db, fname, backend):
    r = root_metrics.compositions(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'Compositions' == r.name


@dbtest
def test_recipes(db, fname, backend):
    r = root_metrics.recipes(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'Recipes' == r.name


@dbtest
def test_products(db, fname, backend):
    r = root_metrics.products(db=db)
    obs = r()
    if obs is None:
        return
    assert 0 < len(obs)
    assert 'Products' == r.name


@dbtest
def test_res_creators(db, fname, backend):
    r = root_metrics.res_creators(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'ResCreators' == r.name


@dbtest
def test_agent_entry(db, fname, backend):
    r = root_metrics.agent_entry(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'AgentEntry' == r.name


@dbtest
def test_agent_exit(db, fname, backend):
    r = root_metrics.agent_exit(db=db)
    obs = r()
    if obs is None:
        return
    assert 0 < len(obs)
    assert 'AgentExit' == r.name


@dbtest
def test_transactions(db, fname, backend):
    r = root_metrics.transactions(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'Transactions' == r.name


@dbtest
def test_info(db, fname, backend):
    r = root_metrics.info(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'Info' == r.name


@dbtest
def test_finish(db, fname, backend):
    r = root_metrics.finish(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'Finish' == r.name


@dbtest
def test_input_files(db, fname, backend):
    r = root_metrics.input_files(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'InputFiles' == r.name


@dbtest
def test_decom_schedule(db, fname, backend):
    r = root_metrics.decom_schedule(db=db)
    obs = r()
    if obs is None:
        return
    assert 0 == len(obs)
    assert 'DecomSchedule' == r.name


@dbtest
def test_build_schedule(db, fname, backend):
    r = root_metrics.build_schedule(db=db)
    obs = r()
    assert 0 == len(obs)
    assert 'BuildSchedule' == r.name


@dbtest
def test_snapshots(db, fname, backend):
    r = root_metrics.snapshots(db=db)
    obs = r()
    assert 0 == len(obs)
    assert 'Snapshots' == r.name


@dbtest
def test_inventories(db, fname, backend):
    r = root_metrics.explicit_inventory(db=db)
    obs = r()
    assert 0 == len(obs)
    assert 'ExplicitInventory' == r.name


@dbtest
def test_inventories_compact(db, fname, backend):
    r = root_metrics.explicit_inventory_compact(db=db)
    obs = r()
    assert 0 == len(obs)
    assert 'ExplicitInventoryCompact' == r.name


@dbtest
def test_resources_non_existent_filter(db, fname, backend):
    r = root_metrics.resources(db=db)
    obs = r(conds=[('NotAColumn', '!=', 'not-a-value')])
    assert 0 == len(obs)
    assert 'Resources' == r.name
