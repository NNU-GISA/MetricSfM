# This is an example of a LocalConfig.cmake file. This is useful on Windows
# platforms, where CMake cannot find many of the libraries automatically and
# it is tedious to set their paths manually in the GUI. To use this file, copy
# and rename it to LocalConfig.cmake

set(MY_BOOST_PATH "F:/DevelopCenter/ThirdParty/boost_1_65_0") # change the path to yout fold of boost
set(MY_OPENCV_PATH "F:/DevelopCenter/ThirdParty/opencv-2.4.13.6/build")  # change the path that your fold containing OpenCVConfig.cmake

message(STATUS "NOTE: Using LocalConfig.cmake to override CMake settings in the GUI")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")

set(OpenCV_DIR ${MY_OPENCV_PATH}) 

set(EIGEN3_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty/Eigen3 CACHE PATH "" FORCE)

set(BOOST_ROOT ${MY_BOOST_PATH} CACHE PATH "" FORCE)
set(BOOST_LIBRARYDIR ${MY_BOOST_PATH}/lib64-msvc-14.1 CACHE PATH "" FORCE)

set(OPENGL_gl_LIBRARY "opengl32" CACHE STRING "" FORCE)
set(OPENGL_glu_LIBRARY "glu32" CACHE STRING "" FORCE)

#set(GLEW_INCLUDE_DIR_HINTS ${PROJECT_SOURCE_DIR}/thirdparty/glew-1.5.1/include CACHE PATH "" FORCE)
#set(GLEW_LIBRARY_DIR_HINTS ${PROJECT_SOURCE_DIR}/thirdparty/glew-1.5.1/lib CACHE FILEPATH "" FORCE)

set(GLOG_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty/glog/include CACHE PATH "" FORCE)
set(GLOG_LIBRARIES_DEBUG ${PROJECT_SOURCE_DIR}/thirdparty/glog/lib/glogd.lib CACHE FILEPATH "" FORCE)
set(GLOG_LIBRARIES_RELEASE ${PROJECT_SOURCE_DIR}/thirdparty/glog/lib/glogr.lib CACHE FILEPATH "" FORCE)

set(GFLAGS_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty/gflags/include CACHE PATH "" FORCE)
set(GFLAGS_LIBRARY_DEBUG_ONE ${PROJECT_SOURCE_DIR}/thirdparty/gflags/lib/gflagsd.lib CACHE FILEPATH "" FORCE)
set(GFLAGS_LIBRARY_DEBUG_TWO ${PROJECT_SOURCE_DIR}/thirdparty/gflags/lib/gflags_nothreadd.lib CACHE FILEPATH "" FORCE)
set(GFLAGS_LIBRARY_RELEASE_ONE ${PROJECT_SOURCE_DIR}/thirdparty/gflags/lib/gflagsr.lib CACHE FILEPATH "" FORCE)
set(GFLAGS_LIBRARY_RELEASE_TWO ${PROJECT_SOURCE_DIR}/thirdparty/gflags/lib/gflags_nothreadr.lib CACHE FILEPATH "" FORCE)

set(CERES_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty/ceres-1.13/include CACHE PATH "" FORCE)
set(CERES_LIBRARIES_DEBUG ${PROJECT_SOURCE_DIR}/thirdparty/ceres-1.13/lib/ceresd.lib CACHE PATH "" FORCE)
set(CERES_LIBRARIES_RELEASE ${PROJECT_SOURCE_DIR}/thirdparty/ceres-1.13/lib/ceresr.lib CACHE PATH "" FORCE)

set(FBOW_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty/fbow/include CACHE PATH "" FORCE)
set(FBOW_LIBRARIES_DEBUG ${PROJECT_SOURCE_DIR}/thirdparty/fbow/lib/fbow001d.lib CACHE PATH "" FORCE)
set(FBOW_LIBRARIES_RELEASE ${PROJECT_SOURCE_DIR}/thirdparty/fbow/lib/fbow001r.lib CACHE PATH "" FORCE)

set(VLFEAT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty/vlfeat/include CACHE PATH "" FORCE)
set(VLFEAT_LIBRARIES_DEBUG ${PROJECT_SOURCE_DIR}/thirdparty/vlfeat/lib/vlfeatd.lib CACHE PATH "" FORCE)
set(VLFEAT_LIBRARIES_RELEASE ${PROJECT_SOURCE_DIR}/thirdparty/vlfeat/lib/vlfeatr.lib CACHE PATH "" FORCE)

set(CUDASIFT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty/cudasift/include CACHE PATH "" FORCE)
set(CUDASIFT_LIBRARIES_DEBUG ${PROJECT_SOURCE_DIR}/thirdparty/cudasift/lib/cudaSiftd.lib CACHE PATH "" FORCE)
set(CUDASIFT_LIBRARIES_RELEASE ${PROJECT_SOURCE_DIR}/thirdparty/cudasift/lib/cudaSiftr.lib CACHE PATH "" FORCE)

set(FLANN_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty/flann/include CACHE PATH "" FORCE)
set(FLANN_LIBRARIES_DEBUG ${PROJECT_SOURCE_DIR}/thirdparty/flann/lib/flannd.lib CACHE PATH "" FORCE)
set(FLANN_LIBRARIES_RELEASE ${PROJECT_SOURCE_DIR}/thirdparty/flann/lib/flannr.lib CACHE PATH "" FORCE)

set(GDAL_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty/gdal/include CACHE PATH "" FORCE)
set(GDAL_LIBRARIES_DEBUG ${PROJECT_SOURCE_DIR}/thirdparty/gdal/lib/package_gdal_x64.lib CACHE PATH "" FORCE)
set(GDAL_LIBRARIES_RELEASE ${PROJECT_SOURCE_DIR}/thirdparty/gdal/lib/package_gdal_x64.lib CACHE PATH "" FORCE)

#set(VLFEAT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty/vlfeat CACHE PATH "" FORCE)
