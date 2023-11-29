"""Tests for root metrics"""
from __future__ import print_function, unicode_literals

from cymetric import root_metrics

from tools import dbtest


def test_resources(dbtest):
    db, fname, backend = dbtest
    r = root_metrics.resources(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'Resources' == r.name


def test_compositions(dbtest):
    db, fname, backend = dbtest
    r = root_metrics.compositions(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'Compositions' == r.name


def test_recipes(dbtest):
    db, fname, backend = dbtest
    r = root_metrics.recipes(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'Recipes' == r.name


def test_products(dbtest):
    db, fname, backend = dbtest
    r = root_metrics.products(db=db)
    obs = r()
    if obs is None:
        return
    assert 0 < len(obs)
    assert 'Products' == r.name


def test_res_creators(dbtest):
    db, fname, backend = dbtest
    r = root_metrics.res_creators(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'ResCreators' == r.name


def test_agent_entry(dbtest):
    db, fname, backend = dbtest
    r = root_metrics.agent_entry(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'AgentEntry' == r.name


def test_agent_exit(dbtest):
    db, fname, backend = dbtest
    r = root_metrics.agent_exit(db=db)
    obs = r()
    if obs is None:
        return
    assert 0 < len(obs)
    assert 'AgentExit' == r.name


def test_transactions(dbtest):
    db, fname, backend = dbtest
    r = root_metrics.transactions(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'Transactions' == r.name


def test_info(dbtest):
    db, fname, backend = dbtest
    r = root_metrics.info(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'Info' == r.name


def test_finish(dbtest):
    db, fname, backend = dbtest
    r = root_metrics.finish(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'Finish' == r.name


def test_input_files(dbtest):
    db, fname, backend = dbtest
    r = root_metrics.input_files(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'InputFiles' == r.name


def test_decom_schedule(dbtest):
    db, fname, backend = dbtest
    r = root_metrics.decom_schedule(db=db)
    obs = r()
    if obs is None:
        return
    assert 0 < len(obs)
    assert 'DecomSchedule' == r.name


def test_build_schedule(dbtest):
    db, fname, backend = dbtest
    r = root_metrics.build_schedule(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'BuildSchedule' == r.name


def test_snapshots(dbtest):
    db, fname, backend = dbtest
    r = root_metrics.snapshots(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'Snapshots' == r.name


def test_inventories(dbtest):
    db, fname, backend = dbtest
    r = root_metrics.explicit_inventory(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'ExplicitInventory' == r.name


def test_inventories_compact(dbtest):
    db, fname, backend = dbtest
    r = root_metrics.explicit_inventory_compact(db=db)
    obs = r()
    assert 0 < len(obs)
    assert 'ExplicitInventoryCompact' == r.name


def test_resources_non_existent_filter(dbtest):
    db, fname, backend = dbtest
    r = root_metrics.resources(db=db)
    obs = r(conds=[('NotAColumn', '!=', 'not-a-value')])
    assert 0 < len(obs)
    assert 'Resources' == r.name
