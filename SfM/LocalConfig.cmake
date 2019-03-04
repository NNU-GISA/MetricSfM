# This is an example of a LocalConfig.cmake file. This is useful on Windows
# platforms, where CMake cannot find many of the libraries automatically and
# it is tedious to set their paths manually in the GUI. To use this file, copy
# and rename it to LocalConfig.cmake
message(STATUS "NOTE: Using LocalConfig.cmake to override CMake settings in the GUI")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")

set(EIGEN3_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty/Eigen3 CACHE PATH "" FORCE)

set(VLFEAT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty/vlfeat/include CACHE PATH "" FORCE)
set(VLFEAT_LIBRARIES_DEBUG ${PROJECT_SOURCE_DIR}/thirdparty/vlfeat/lib/vlfeatd.lib CACHE PATH "" FORCE)
set(VLFEAT_LIBRARIES_RELEASE ${PROJECT_SOURCE_DIR}/thirdparty/vlfeat/lib/vlfeatr.lib CACHE PATH "" FORCE)

set(CUDASIFT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty/cudasift/include CACHE PATH "" FORCE)
set(CUDASIFT_LIBRARIES_DEBUG ${PROJECT_SOURCE_DIR}/thirdparty/cudasift/lib/cudaSiftd.lib CACHE PATH "" FORCE)
set(CUDASIFT_LIBRARIES_RELEASE ${PROJECT_SOURCE_DIR}/thirdparty/cudasift/lib/cudaSiftr.lib CACHE PATH "" FORCE)

set(SIFTGPU_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty/siftgpu/include CACHE PATH "" FORCE)
set(SIFTGPU_LIBRARIES_DEBUG ${PROJECT_SOURCE_DIR}/thirdparty/siftgpu/lib/siftgpud.lib CACHE PATH "" FORCE)
set(SIFTGPU_LIBRARIES_RELEASE ${PROJECT_SOURCE_DIR}/thirdparty/siftgpu/lib/siftgpur.lib CACHE PATH "" FORCE)

# set boost 
#set(MY_BOOST_PATH "F:/DevelopCenter/ThirdParty/boost_1_65_0") # change the path to yout fold of boost
#set(BOOST_ROOT ${MY_BOOST_PATH} CACHE PATH "" FORCE)
#set(BOOST_LIBRARYDIR ${MY_BOOST_PATH}/lib64-msvc-14.1 CACHE PATH "" FORCE)

# fbow
set(FBOW_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty/fbow/include CACHE PATH "" FORCE)
set(FBOW_LIBRARIES_DEBUG ${PROJECT_SOURCE_DIR}/thirdparty/fbow/lib/fbow001d.lib CACHE PATH "" FORCE)
set(FBOW_LIBRARIES_RELEASE ${PROJECT_SOURCE_DIR}/thirdparty/fbow/lib/fbow001r.lib CACHE PATH "" FORCE)