cmake_minimum_required(VERSION 3.0.2)
project(handsome_test)

find_package(catkin REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED PATHS "/code/lib/libtorch_cpu/libtorch")
set(CMAKE_CXX_STANDARD 17)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(main main.cpp)
target_include_directories(main PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(main PUBLIC ${TORCH_LIBRARIES})



#############
## Install ##
#############


## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
install(TARGETS main
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
