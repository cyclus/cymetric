# Python extension modules
include_directories(${CYMETRIC_INCLUDE_DIRS})

# Print include dir
get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("-- Include paths for ${CMAKE_CURRENT_SOURCE_DIR}: ${inc_dirs}")


# nucname
#set_source_files_properties("${PROJECT_SOURCE_DIR}/pyne/nucname.pyx"
#                            PROPERTIES CYTHON_IS_CXX TRUE)
#cython_add_module(nucname nucname.pyx)
#target_link_libraries(nucname pyne)

