""" Convert able to the default unit system.
"""

try:
    from cymetric import schemas
    from cymetric import tools
    from cymetric import evaluator
except ImportError:
    # some wacky CI paths prevent absolute importing, try relative
    from . import schemas
    from . import tools
    from . import evaluator

import pint

ureg = pint.UnitRegistry()


def build_normalized_metric(raw_metric):

    _norm_deps = [raw_metric.__name__]

    _norm_schema = evaluator.build_normalized_metric(raw_metric)
    _norm_name = "norm_" + raw_metric.__name__

    @metric(name=_norm_name, depends=_norm_deps, schema=_norm_schema)
      def norm_metric(raw):
           if (raw_cls.__name__ not in UNITS_REGISTRY):
                return raw

            unit_registry = evaluator.UNITS_REGISTRY[raw.__name__]

            for key in unit_registry:
                for unit in unit_registry[key]:
                    u_col_name = unit[0]
                    conv = [ureg.parse_expression(x).to_root_units().magnitude
                            for x in raw[u_col_name]]

            return z

    del _matdeps, _matschema


def build_conversion_col(unit_col):
    conversion_col = ureg.parse_expression(unit_col)
    return conversion_col
