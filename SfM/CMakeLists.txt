cmake_minimum_required(VERSION 2.8)
project(MetricSfM)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

ADD_DEFINITIONS(-DGLOG_NO_ABBREVIATED_SEVERITIES)
ADD_DEFINITIONS(-DNOMINMAX)

if(WIN32)
    add_definitions(/DGOOGLE_GLOG_DLL_DECL=)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(${CMAKE_SOURCE_DIR}/cmake/CheckSSEExtensions.cmake)

if(EXISTS ${CMAKE_SOURCE_DIR}/LocalConfig.cmake)
    include(${CMAKE_SOURCE_DIR}/LocalConfig.cmake)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(THIRDPARTY_ROOT_FOLDER "thirdparty")
set(TEST_ROOT_FOLDER "test")
set(PROJECT_ROOT_DIR ${PROJECT_SOURCE_DIR})

# CUDA
find_package(CUDA REQUIRED)
#list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-O2")
list(APPEND CUDA_NVCC_FLAGS "--generate-code arch=compute_61,code=sm_61")

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# OpenCV
find_package(OpenCV REQUIRED)

# Boost
# refer to https://stackoverflow.com/questions/28887680/linking-boost-library-with-boost-use-static-lib-off-on-windows
set(Boost_NO_SYSTEM_PATHS true)
set(Boost_USE_STATIC_LIBS OFF CACHE BOOL "use static libraries from Boost")
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS program_options thread filesystem regex system unit_test_framework REQUIRED)
if (WIN32)
  # disable autolinking in boost
  add_definitions( -DBOOST_ALL_NO_LIB )
endif()

# Eigen
find_package(Eigen3 REQUIRED)

# Generate lib for current project
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/src
${PROJECT_SOURCE_DIR}/thirdparty
${OpenCV_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIRS}
${GLOG_INCLUDE_DIRS}
${CERES_INCLUDE_DIRS}
${FBOW_INCLUDE_DIRS}
${VLFEAT_INCLUDE_DIRS}
${CUDASIFT_INCLUDE_DIRS}
${FLANN_INCLUDE_DIRS}
${GDAL_INCLUDE_DIRS}
)

FILE(GLOB OBJ_HDRS  "src/*.h" "src/utils/*.hpp")
FILE(GLOB OBJ_SRCS  "src/*.c" "src/*.cc" "src/*.cpp")
source_group(src FILES ${OBJ_HDRS} ${OBJ_SRCS})

FILE(GLOB Feature_HDRS  "src/feature/*.h" "src/utils/*.hpp")
FILE(GLOB Feature_SRCS  "src/feature/*.c" "src/feature/*.cc" "src/feature/*.cpp")
source_group(feature FILES ${Feature_HDRS} ${Feature_SRCS})

FILE(GLOB Graph_HDRS  "src/graph/*.h" "src/utils/*.hpp")
FILE(GLOB Graph_SRCS  "src/graph/*.c" "src/graph/*.cc" "src/graph/*.cpp")
source_group(graph FILES ${Graph_HDRS} ${Graph_SRCS})

FILE(GLOB Orientation_HDRS  "src/orientation/*.h" "src/utils/*.hpp")
FILE(GLOB Orientation_SRCS  "src/orientation/*.c" "src/orientation/*.cc" "src/graph/*.cpp")
source_group(orientation FILES ${Orientation_HDRS} ${Orientation_SRCS})

FILE(GLOB Utils_HDRS  "src/utils/*.h" "src/utils/*.hpp")
FILE(GLOB Utils_SRCS  "src/utils/*.c" "src/utils/*.cc" "src/utils/*.cpp")
source_group(utils FILES ${Utils_HDRS} ${Utils_SRCS})

cuda_add_library(${PROJECT_NAME} 
STATIC 
${OBJ_HDRS}          ${OBJ_SRCS} 
${Feature_HDRS}      ${Feature_SRCS}
${Graph_HDRS}        ${Graph_SRCS}
${Orientation_HDRS}  ${Orientation_SRCS}
${Utils_HDRS}        ${Utils_SRCS}
)

target_link_libraries(${PROJECT_NAME}
legacy_stdio_definitions.lib
shlwapi.lib
${CMAKE_DL_LIBS}
${OpenCV_LIBS}
${Boost_LIBRARIES}
)

target_link_libraries(${PROJECT_NAME} debug ${FBOW_LIBRARIES_DEBUG} optimized ${FBOW_LIBRARIES_RELEASE})
target_link_libraries(${PROJECT_NAME} debug ${GFLAGS_LIBRARY_DEBUG_ONE} optimized ${GFLAGS_LIBRARY_RELEASE_ONE})
target_link_libraries(${PROJECT_NAME} debug ${GFLAGS_LIBRARY_DEBUG_TWO} optimized ${GFLAGS_LIBRARY_RELEASE_TWO})
target_link_libraries(${PROJECT_NAME} debug ${GLOG_LIBRARIES_DEBUG} optimized ${GLOG_LIBRARIES_RELEASE})
target_link_libraries(${PROJECT_NAME} debug ${CERES_LIBRARIES_DEBUG} optimized ${CERES_LIBRARIES_RELEASE})
target_link_libraries(${PROJECT_NAME} debug ${VLFEAT_LIBRARIES_DEBUG} optimized ${VLFEAT_LIBRARIES_RELEASE})
target_link_libraries(${PROJECT_NAME} debug ${CUDASIFT_LIBRARIES_DEBUG} optimized ${CUDASIFT_LIBRARIES_RELEASE})
target_link_libraries(${PROJECT_NAME} debug ${FLANN_LIBRARIES_DEBUG} optimized ${FLANN_LIBRARIES_RELEASE})
target_link_libraries(${PROJECT_NAME} debug ${GDAL_LIBRARIES_DEBUG} optimized ${GDAL_LIBRARIES_RELEASE})

add_subdirectory(test)
