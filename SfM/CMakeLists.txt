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
set(OpenCV_DIR "F:/DevelopCenter/ThirdParty/opencv-2.4.13.6/build")  # change the path that your fold containing OpenCVConfig.cmake
find_package(OpenCV REQUIRED)

# Ceres
find_package(Ceres CONFIG REQUIRED)

# Eigen3
find_package(Eigen3 CONFIG REQUIRED)

# flann
find_package(flann REQUIRED)

# GLUT
find_package(GLUT REQUIRED)

# OpenGL
find_package(OpenGL REQUIRED)

# Glew
find_package(Glew)

# Generate lib for current project
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/src
${PROJECT_SOURCE_DIR}/src/dense/cudasgm/include
${PROJECT_SOURCE_DIR}/src/dense/libelas/include
${PROJECT_SOURCE_DIR}/thirdparty
${OpenCV_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIRS}
${CERES_INCLUDE_DIRS}
${VLFEAT_INCLUDE_DIRS}
${CUDASIFT_INCLUDE_DIRS}
${SIFTGPU_INCLUDE_DIRS}
${FBOW_INCLUDE_DIRS}
)

FILE(GLOB Utils_HDRS  "src/utils/*.h" "src/utils/*.hpp")
FILE(GLOB Utils_SRCS  "src/utils/*.c" "src/utils/*.cc" "src/utils/*.cpp")
source_group(utils FILES ${Utils_HDRS} ${Utils_SRCS})

SET(Basic_Files  
src/basic_structs.h
src/camera.h src/camera.cc
src/database.h src/database.cc
src/graph.h src/graph.cc
src/optimizer.h src/optimizer.cc
src/structure.h src/structure.cc
src/accuracy_accessment.h src/accuracy_accessment.cc)

FILE(GLOB Feature_HDRS  "src/feature/*.h" "src/feature/*.hpp")
FILE(GLOB Feature_SRCS  "src/feature/*.c" "src/feature/*.cc" "src/feature/*.cpp")
source_group(feature FILES ${Feature_HDRS} ${Feature_SRCS})
FILE(GLOB Graph_HDRS  "src/graph/*.h" "src/graph/*.hpp")
FILE(GLOB Graph_SRCS  "src/graph/*.c" "src/graph/*.cc" "src/graph/*.cpp")
source_group(graph FILES ${Graph_HDRS} ${Graph_SRCS})
FILE(GLOB Orientation_HDRS  "src/orientation/*.h" "src/orientation/*.hpp")
FILE(GLOB Orientation_SRCS  "src/orientation/*.c" "src/orientation/*.cc" "src/orientation/*.cpp")
source_group(orientation FILES ${Orientation_HDRS} ${Orientation_SRCS})

#### SfM
SET(SFM_Files src/sfm_incremental.h src/sfm_incremental.cc)

## SLAM
SET(SLAM_Files src/slam_gps.h src/slam_gps.cc)

cuda_add_library(SFMLib 
STATIC 
${Basic_Files}
${SFM_Files}
${SLAM_Files}
${Feature_HDRS}      ${Feature_SRCS}
${Graph_HDRS}        ${Graph_SRCS}
${Orientation_HDRS}  ${Orientation_SRCS}
${Utils_HDRS}        ${Utils_SRCS}
)
target_link_libraries(SFMLib legacy_stdio_definitions.lib shlwapi.lib
${CMAKE_DL_LIBS} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${FLANN_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARIES}) #${GLOG_LIBRARIES} 
target_link_libraries(SFMLib debug ${VLFEAT_LIBRARIES_DEBUG} optimized ${VLFEAT_LIBRARIES_RELEASE})
target_link_libraries(SFMLib debug ${CUDASIFT_LIBRARIES_DEBUG} optimized ${CUDASIFT_LIBRARIES_RELEASE})
target_link_libraries(SFMLib debug ${SIFTGPU_LIBRARIES_DEBUG} optimized ${SIFTGPU_LIBRARIES_RELEASE})
target_link_libraries(SFMLib debug ${FBOW_LIBRARIES_DEBUG} optimized ${FBOW_LIBRARIES_RELEASE})

## CUDA SGM
configure_file(${PROJECT_SOURCE_DIR}/src/dense/cudasgm/include/libsgm_config.h.in
               ${PROJECT_SOURCE_DIR}/src/dense/cudasgm/include/libsgm_config.h )
FILE(GLOB CUDASGM_HDRS  "src/dense/cudasgm/include/*.h")
FILE(GLOB CUDASGM_SRCS  "src/dense/cudasgm/src/*.cpp" "src/dense/cudasgm/src/*.hpp" "src/dense/cudasgm/src/*.cu")
source_group(cudasgm FILES ${CUDASGM_HDRS} ${CUDASGM_SRCS})
cuda_add_library(SGMLib STATIC ${CUDASGM_HDRS} ${CUDASGM_SRCS})
#target_link_libraries(SGMLib ${CMAKE_DL_LIBS} ${OpenCV_LIBS} )

## ELAS
FILE(GLOB ELAS_HDRS  "src/dense/libelas/include/*.h")
FILE(GLOB ELAS_SRCS  "src/dense/libelas/src/*.cpp")
source_group(libelas FILES ${ELAS_HDRS} ${ELAS_SRCS})
add_library(ELASLib STATIC ${ELAS_HDRS} ${ELAS_SRCS})
target_link_libraries(ELASLib ${CMAKE_DL_LIBS} ${OpenCV_LIBS} )

SET(Dense_Files src/dense_reconstruction.h src/dense_reconstruction.cc)
cuda_add_library(DenseLib STATIC ${Basic_Files} ${Dense_Files})
target_link_libraries(DenseLib ${CMAKE_DL_LIBS} ${OpenCV_LIBS} ${SGMLib} ${ELASLib} )

add_subdirectory(test)
