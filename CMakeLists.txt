cmake_minimum_required( VERSION 3.1 )
project( joinMap )

set( CMAKE_BUILD_TYPE Release )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# opencv 
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# eigen 
include_directories( "/usr/include/eigen3/" )

# octomap 
find_package( octomap REQUIRED )
include_directories( ${OCTOMAP_INCLUDE_DIRS} )


add_executable( joinMap joinMap.cpp )
target_link_libraries( joinMap ${OpenCV_LIBS} )

add_executable( octoMap octoMap.cpp )
target_link_libraries( octoMap ${OpenCV_LIBS} ${OCTOMAP_LIBRARIES} )


