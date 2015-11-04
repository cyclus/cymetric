from cymetric import metric

#########################
## FCO-related metrics ##
#########################

# Electricity Generated [GWe-y]
_egdeps = [('ElectricityGenerated', ('AgentId', 'Year'), 'Power'),]

_egschema = [('Year', ts.INT), ('Power', ts.DOUBLE)]

@metric(name='FcoElectricityGenerated', depends=_egdeps, schema=_egschema)
def fco_electricity_generated(series):
    """FcoElectricityGenerated metric returns the electricity generated in GWe-y 
    for all agents in simulation.
    """
    elec = series[0].reset_index()
    elec = elec.groupby('AgentId').sum()
    elec = pd.DataFrame(data={'Year': elec.Year, 
                              'Power': elec.Value.apply(lambda x: x/1000)}, 
                        columns=['Year', 'Power'])
    rtn = elec.reset_index()
    return rtn

del _egdeps, _egschema


# U Resources Mined [t] 
_udeps= [
    ('Materials', ('ResourceId', 'ObjId', 'TimeCreated', 'NucId'), 'Mass'),
    ('Transactions', ('ResourceId', ), 'Commodity')
    ]

_uschema = [('Year', ts.INT), ('UMined', ts.DOUBLE)]

@metric(name='FcoUMined', depends=_udeps, schema=_uschema)
def fco_u_mined(series):
    """FcoUMined metric returns the uranium mined in tonnes for each year 
    in a 200-yr simulation. This is written for FCO databases that use the 
    Bright-lite Fuel Fab(i.e., the U235 and U238 are given separately in the 
    FCO simulations).
    """
    tools.raise_no_pyne('U_Mined could not be computed', HAVE_PYNE)
    mass = pd.merge(series[0].reset_index(), series[1].reset_index(), 
            on=['ResourceId'], how='inner').set_index(['ObjId', 
                'TimeCreated', 'NucId'])
    u = []
    prods = {}
    mass235 = {}
    m = mass[mass['Commodity'] == 'LWR Fuel']
    for (obj, _, nuc), value in m.iterrows():
        if 922320000 <= nuc <= 922390000:
            prods[obj] = prods.get(obj, 0.0) + value['Mass']
        if nuc==922350000:
            mass235[obj] = value['Mass']
    x_feed = 0.0072
    x_tails = 0.0025
    for obj, m235 in mass235.items():
        x_prod = m235 / prods[obj]
        feed = enr.feed(x_feed, x_prod, x_tails, product=prods[obj]) / 1000
        u.append(feed)
    m = m.groupby(level=['ObjId', 'TimeCreated'])['Mass'].sum()
    m = m.reset_index()
    # sum by years (12 time steps)
    u = pd.DataFrame(data={'Year': m.TimeCreated.apply(lambda x: x//12), 
                           'UMined': u}, columns=['Year', 'UMined'])
    u = u.groupby('Year').sum()
    rtn = u.reset_index()
    return rtn

del _udeps, _uschema


# SWU Required [million SWU]
_swudeps = [
    ('Materials', ('ResourceId', 'ObjId', 'TimeCreated', 'NucId'), 'Mass'),
    ('Transactions', ('ResourceId',), 'Commodity')
    ]

_swuschema = [('Year', ts.INT), ('SWU', ts.DOUBLE)]

@metric(name='FcoSwu', depends=_swudeps, schema=_swuschema)
def fco_swu(series):
    """FcoSwu metric returns the separative work units required for each 
    year in a 200-yr simulation. This is written for FCO databases that 
    use the Bright-lite (i.e., the U235 and U238 are given separately 
    in the FCO simulations).
    """
    tools.raise_no_pyne('SWU Required could not be computed', HAVE_PYNE)
    mass = pd.merge(series[0].reset_index(), series[1].reset_index(),
            on=['ResourceId'], how='inner').set_index(['ObjId', 'TimeCreated', 'NucId'])
    swu = []
    prods = {}
    mass235 = {}
    m = mass[mass['Commodity'] == 'LWR Fuel']
    for (obj, _, nuc), value in m.iterrows():
        if 922320000 <= nuc <= 922390000:
            prods[obj] = prods.get(obj, 0.0) + value['Mass']
        if nuc == 922350000:
            mass235[obj] = value['Mass']
    x_feed = 0.0072
    x_tails = 0.0025
    for obj, m235 in mass235.items():
        x_prod = m235 / prods[obj]
        swu0 = enr.swu(x_feed, x_prod, x_tails, product=prods[obj]) / 1e6
        swu.append(swu0)
    m = m.groupby(level=['ObjId', 'TimeCreated'])['Mass'].sum()
    m = m.reset_index()
    # sum by years (12 time steps)
    swu = pd.DataFrame(data={'Year': m.TimeCreated.apply(lambda x: x//12),
                             'SWU': swu}, columns=['Year', 'SWU'])
    swu = swu.groupby('Year').sum()
    rtn = swu.reset_index()
    return rtn

del _swudeps, _swuschema


# Annual Fuel Loading Rate [tHM/y]
_fldeps = [
    ('Materials', ('ResourceId', 'TimeCreated'), 'Mass'),
    ('Transactions', ('ResourceId',), 'Commodity')
    ]

_flschema = [('Year', ts.INT), ('FuelLoading', ts.DOUBLE)]

@metric(name='FcoFuelLoading', depends=_fldeps, schema=_flschema)
def fco_fuel_loading(series):
    """FcoFuelLoading metric returns the fuel loaded in tHM/y in a 200-yr 
    simulation. This is written for FCO databases.
    """
    mass = pd.merge(series[0].reset_index(), series[1].reset_index(),
            on=['ResourceId'], how='inner').set_index(['TimeCreated'])
    mass = mass.query('Commodity == ["LWR Fuel", "FR Fuel"]')
    mass = mass.groupby(mass.index)['Mass'].sum()
    mass = mass.reset_index()
    # sum by years (12 time steps)
    mass = pd.DataFrame(data={'Year': mass.TimeCreated.apply(lambda x: x//12),
                              'FuelLoading': mass.Mass.apply(lambda x: x/1000)}, 
                        columns=['Year', 'FuelLoading'])
    mass = mass.groupby('Year').sum()
    rtn = mass.reset_index()
    return rtn

del _fldeps, _flschema


