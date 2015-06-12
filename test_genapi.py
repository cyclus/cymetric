import nose
from nose.tools import assert_equal

import genapi


def test_enumtypes():
    test = (
        ('std::map< int,std::vector<double> >', 
         set(['MAP_INT_VECTOR_DOUBLE', 
              'VL_MAP_INT_VECTOR_DOUBLE',
              'MAP_INT_VL_VECTOR_DOUBLE',
              'VL_MAP_INT_VL_VECTOR_DOUBLE',]),
         2,),
        )
    for obj, exp_t, exp_r in test:
        yield assert_equal, exp_t, set(genapi.enumtypes(obj))
        yield assert_equal, exp_r, genapi.rank(obj)
