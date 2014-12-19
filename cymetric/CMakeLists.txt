# Python extension modules
include_directories(${CYMETRIC_INCLUDE_DIRS})

# Print include dir
get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("-- Include paths for ${CMAKE_CURRENT_SOURCE_DIR}: ${inc_dirs}")

# cyclus
set_source_files_properties("${PROJECT_SOURCE_DIR}/cyclus/cyclus.pyx"
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(_cyclus cyclus.pyx)
target_link_libraries(_cyclus ${LIBS})
set_target_properties(_cyclus PROPERTIES OUTPUT_NAME cyclus LINKER_LANGUAGE CXX)

