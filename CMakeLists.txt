# Defines the CMake commands/policies
cmake_minimum_required( VERSION 2.8.5 )

# Set the project name
project(cymetric)

# Make the scripts available in the 'cmake' directory available for the
# 'include()' command, 'find_package()' command.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Default to release build type
set(CMAKE_BUILD_TYPE Release CACHE STRING "The build type" FORCE)

# Also, there are some custom macros to make for less verbosity
include( CymetricMacros )
print_logo()

# quiets fortify_source warnings when not compiling with optimizations
# in linux distros where compilers were compiled with fortify_source enabled by
# default (e.g. Arch linux).
MESSAGE("-- Build type: ${CMAKE_BUILD_TYPE}")
STRING(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_TYPE)
IF(NOT ${BUILD_TYPE} STREQUAL "release")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0")
ENDIF()

# no overflow warnings because of silly coin-ness
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-overflow")

#
# Find and use dependencues
#

# Find Cyclus, it is kind of a big deal
find_package(Cyclus REQUIRED)
SET(CYMETRIC_INCLUDE_DIR ${CYMETRIC_INCLUDE_DIR} ${CYCLUS_CORE_INCLUDE_DIR})
SET(LIBS ${LIBS} ${CYCLUS_CORE_LIBRARIES})

# Use new Python library finder
find_package(PythonInterp REQUIRED)
find_package(PythonLibsNew REQUIRED)
message("-- PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")

# Include the CMake script UseCython.cmake.  This defines add_cython_module().
# Instruction for use can be found at the top of cmake/UseCython.cmake.
include( UseCython )

execute_process(COMMAND "${CYTHON_EXECUTABLE}" "-V"
    ERROR_VARIABLE CYTHON_VERSION
    ERROR_STRIP_TRAILING_WHITESPACE)
message( STATUS "Cython Version: " ${CYTHON_VERSION} )

# find numpy and include the numpy headers
find_package(Numpy REQUIRED)
SET(CYMETRIC_INCLUDE_DIR "${NUMPY_INCLUDE_DIRS}" ${CYMETRIC_INCLUDE_DIR})

# Include the accumlated directory
INCLUDE_DIRECTORIES(${CYMETRIC_INCLUDE_DIR})

#
# Continue with build settings
#

# This makes all the libraries build as SHARED
set(BUILD_SHARED_LIBS true)
message("-- CMake Install Prefix: ${CMAKE_INSTALL_PREFIX}")

# RPATH Settings
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if(APPLE)
    # I think that this is the right thing to do for MacOSX 10.5+ --Anthony
    set(CMAKE_MACOSX_RPATH 1)
    set(CMAKE_INSTALL_RPATH "@loader_path/../../..")
elseif(WIN32)
    if(MSVC)
        # ??? Who knows what to do here?! --Anthony
    elseif(CMAKE_COMPILER_IS_GNUC OR CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_INSTALL_RPATH "\$ORIGIN/lib")
    endif(MSVC)
else(APPLE)
    # For linux
    set(CMAKE_INSTALL_RPATH "\$ORIGIN/lib")
endif(APPLE)
message("-- RPATH: ${CMAKE_INSTALL_RPATH}")

# With CMake, a clean separation can be made between the source tree and the
# build tree.  When all source is compiled, as with pure C/C++, the source is
# no-longer needed in the build tree.  However, with pure *.py source, the
# source is processed directly.  To handle this, we reproduce the availability
# of the source files in the build tree.
add_custom_target(ReplicatePythonSourceTree ALL "${CMAKE_COMMAND}" -P
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ReplicatePythonSourceTree.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(cymetric)

# Print include dir
get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("-- Include paths for ${CMAKE_CURRENT_SOURCE_DIR}: ${inc_dirs}")

message("-- Copying C/C++ header files.")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/cpp/" DESTINATION
    "${CMAKE_BINARY_DIR}/cymetric/include/" FILES_MATCHING PATTERN "*.h")

message("-- Copying Cython header files.")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/cymetric/"
    DESTINATION "${CMAKE_BINARY_DIR}/cymetric/"
    FILES_MATCHING PATTERN "*.pxd"
                   PATTERN "lib" EXCLUDE
                   PATTERN "include" EXCLUDE)

message("-- Copying scripts.")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/scripts" DESTINATION "${CMAKE_BINARY_DIR}")
