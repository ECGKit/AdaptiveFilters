project(LMS)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_BUILD_TYPE RELEASE)
set(SRC ".")
set(INCLUDE_DIR "../../include")
set(EIGEN_HOME "../../include/eigen3/Eigen")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")

#find_package(Boost 1.36.0 COMPONENTS iostreams system filesystem)
find_package(Boost 1.36.0 COMPONENTS system)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${INCLUDE_DIR})
include_directories(${EIGEN_HOME})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
