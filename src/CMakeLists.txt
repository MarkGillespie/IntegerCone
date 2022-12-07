cmake_minimum_required(VERSION 2.8.9)

# Maybe stop from CMAKEing in the wrong place
if (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "Source and build directories cannot be the same. Go use the /build directory.")
endif()

add_library(
  gen-cones
  ConeGene.cpp
  ConesFlattening.cpp
  MeshDefinition.cpp
  MeshLaplace.cpp
  MeshTopo.cpp
  )
target_include_directories(gen-cones PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")

add_executable(
  cones
  main.cpp
)
target_include_directories(gen-cones PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
# target_include_directories(gen-cones PUBLIC ${EIGEN3_INCLUDE_DIR})
target_include_directories(gen-cones PUBLIC ${OPENMESH_INCLUDE_DIR} )
target_include_directories(gen-cones PUBLIC ${OpenMP_CXX_INCLUDE_DIR})
target_include_directories(gen-cones PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../packages")
# # borrow args.hxx directly from polyscope
# target_include_directories(gen-cones PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../deps/polyscope/deps/args/")
target_link_libraries(cones ${OPENMESH_LIBRARIES})
target_link_libraries(cones ${OpenMP_CXX_LIBRARIES})
target_link_libraries(cones gen-cones)


option(PACKAGE_TESTS "Build the tests" OFF)

if(PACKAGE_TESTS)
  enable_testing()
  add_subdirectory(
    tests
  )
endif()
