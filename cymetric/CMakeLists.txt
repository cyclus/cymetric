# Python extension modules
#include_directories(${CYMETRIC_INCLUDE_DIRS} ${INCLUDE_DIRECTORIES})
#include_directories(${CYCLUS_INCLUDE_DIRECTORIES})
#SET(INCLUDE_DIRECTORIES ${INCLUDE_DIRECTORIES} ${CYCLUS_CORE_INCLUDE_DIR})

# Print include dir
get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("-- Include paths for ${CMAKE_CURRENT_SOURCE_DIR}: ${inc_dirs}")

# cyclus
set_source_files_properties("${PROJECT_SOURCE_DIR}/cymetric/cyclus.pyx"
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(cyclus cyclus.pyx)
target_link_libraries(cyclus ${LIBS})
set_target_properties(cyclus PROPERTIES LINKER_LANGUAGE CXX)

