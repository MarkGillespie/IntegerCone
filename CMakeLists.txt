cmake_minimum_required(VERSION 3.10.0)

project(integer_cones)

# == Export compile commands
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

### Configure output locations
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Print the build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release" FORCE)
endif()
message(STATUS "cmake build type: ${CMAKE_BUILD_TYPE}")

### Configure the compiler
# This is a basic, decent setup that should do something sane on most compilers

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

  # using Clang (linux or apple) or GCC
  message("Using clang/gcc compiler flags")
  SET(BASE_CXX_FLAGS "-std=c++17 -Wall -Wextra -g3")
  SET(DISABLED_WARNINGS " -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-deprecated-declarations -Wno-missing-braces -Wno-unused-private-field -Wno-deprecated-copy -Wno-unknown-warning-option")
  SET(TRACE_INCLUDES " -H -Wno-error=unused-command-line-argument")

  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    message("Setting clang-specific options")
    SET(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -ferror-limit=3 -fcolor-diagnostics")
    SET(CMAKE_CXX_FLAGS_DEBUG          "-fsanitize=address -fno-limit-debug-info")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message("Setting gcc-specific options")
    SET(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -fmax-errors=5")
    SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} -Wno-maybe-uninitialized -Wno-format-zero-length -Wno-unused-but-set-parameter -Wno-unused-but-set-variable")
  endif()
  
  SET(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS} ${DISABLED_WARNINGS}")
  SET(CMAKE_CXX_FLAGS_DEBUG          "-g3")
  SET(CMAKE_CXX_FLAGS_RELEASE        "-O3 -march=native -DNDEBUG")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
  message("Using Visual Studio compiler flags")
  set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} /W4")
  set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} /MP") # parallel build
  SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4267\"")  # ignore conversion to smaller type (fires more aggressively than the gcc version, which is annoying)
  SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4244\"")  # ignore conversion to smaller type (fires more aggressively than the gcc version, which is annoying)
  SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4305\"")  # ignore truncation on initialization
  SET(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS} ${DISABLED_WARNINGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

  add_definitions(/D "_CRT_SECURE_NO_WARNINGS")
  add_definitions (-DNOMINMAX)
else()
  # unrecognized
  message( FATAL_ERROR "Unrecognized compiler [${CMAKE_CXX_COMPILER_ID}]" )
endif()


# == Deps


# find_package (Eigen3 3.3 QUIET)

# if(EIGEN3_FOUND)
#   message(STATUS "--    success. find_package() succeeded with Eigen at ${EIGEN3_INCLUDE_DIR}")
# else()
#   message(STATUS "--    Could not find Eigen anywhere.")
# endif()

# include("FindOpenMesh.cmake")
# if(OPENMESH_FOUND)
#   message(STATUS "--    success. find_package() succeeded with OpenMesh at ${OPENMESH_INCLUDE_DIR}")
# else()
#   message(STATUS "--    Could not find OpenMesh anywhere.")
# endif()

find_package(OpenMP REQUIRED)

# add_subdirectory(deps/fcpw)
add_subdirectory(packages/geometry-central)
add_subdirectory(packages/polyscope)

include("findMKL.cmake")
if(MKL_FOUND)
  message("MKL_INCLUDE_DIR: ${MKL_INCLUDE_DIR}")
  message("MKL_LIBRARIES: ${MKL_LIBRARIES}")
  add_definitions(-DEIGEN_USE_MKL_ALL)
  set(GC_HAVE_MKL "TRUE")
else()
  set(GC_HAVE_MKL "FALSE")
endif()
# set(GG_HAVE_MKL ${MKL_FOUND}) # why doesn't this work?
message("MKL_FOUND: ${MKL_FOUND}")
message("GC_HAVE_MKL: ${GC_HAVE_MKL}")

# https://community.intel.com/t5/Intel-C-Compiler/dyld-Library-not-loaded-rpath-libimp5-dylib/m-p/1097284
# Yes, there is a problem on the Mac with @rpath.  Here is a work-around:
# After successfully building your app, on a command line you must use the install_name_tool to relocate libiomp5.dylib in the executable:
# install_name_tool -change @rpath/libiomp5.dylib @executable_path/`basename libiomp5.dylib` <full pathname to your executable>
# where  the full pathname will be something like the following:
# /Users/YourUserName/Developer/YourApp/YourApp.app/Contents/MacOS/YourExecutableAppName
# install_name_tool -change @rpath/libiomp5.dylib /opt/intel/oneapi/compiler/latest/mac/compiler/lib/libiomp5.dylib bin/cones
# ^ didn't seem to work

# this seemed better:
# install_name_tool -add_rpath /opt/intel/oneapi/compiler/latest/mac/compiler/lib/ bin/cones

# or maybe this one:
# install_name_tool -add_rpath /opt/intel/oneapi/compiler/latest/mac/compiler/lib/libiomp5.dylib bin/cones


add_subdirectory(src)
