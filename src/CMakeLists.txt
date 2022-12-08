cmake_minimum_required(VERSION 2.8.9)

# Maybe stop from CMAKEing in the wrong place
if (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "Source and build directories cannot be the same. Go use the /build directory.")
endif()

add_library(
  gen-cones
  # ConeGene.cpp
  ConesFlattening.cpp
  ConfigurableSolver.cpp
  # MeshDefinition.cpp
  # MeshLaplace.cpp
  # MeshTopo.cpp
  )
target_include_directories(gen-cones PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
# target_include_directories(gen-cones PUBLIC ${EIGEN3_INCLUDE_DIR})
# target_include_directories(gen-cones PUBLIC ${OPENMESH_INCLUDE_DIR} )
target_include_directories(gen-cones PUBLIC ${OpenMP_CXX_INCLUDE_DIR})
target_include_directories(gen-cones PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../packages")
# target_link_libraries(gen-cones ${OPENMESH_LIBRARIES})
target_link_libraries(gen-cones ${OpenMP_CXX_LIBRARIES})
target_link_libraries(gen-cones geometry-central polyscope)

if (${GC_HAVE_MKL})
  target_compile_definitions(gen-cones PUBLIC GC_HAVE_MKL)
  message("including MKL headers from: ${MKL_INCLUDE_DIR}")
  target_include_directories(gen-cones PUBLIC "${MKL_INCLUDE_DIR}" )
  target_link_libraries(gen-cones ${MKL_LIBRARIES} )
endif()
message("GC_HAVE_MKL: ${GC_HAVE_MKL}")

if (${GC_HAVE_SUITESPARSE})
  message("including suitesparse headers from: ${GC_SUITESPARSE_INCLUDE_DIRS}")
  target_include_directories(gen-cones PUBLIC ${GC_SUITESPARSE_INCLUDE_DIRS})
  target_link_libraries(gen-cones ${GC_SUITESPARSE_LIBRARIES})
endif()

add_executable(
  cones
  main.cpp
)
target_include_directories(cones PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
# borrow args.hxx directly from polyscope
target_include_directories(cones PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../packages/polyscope/deps/args/")
target_link_libraries(cones gen-cones geometry-central polyscope)


option(PACKAGE_TESTS "Build the tests" OFF)

if(PACKAGE_TESTS)
  enable_testing()
  add_subdirectory(
    tests
  )
endif()
