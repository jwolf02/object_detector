cmake_minimum_required(VERSION 3.1)
project(object_detector)

set(CMAKE_CXX_STANDARD	14)

set(CMAKE_CXX_FLAGS	"${CMAKE_CXX_FLAGS} -Wall -O2 -march=native -mtune=native")

set(OpenCV_DIR "/usr/local")

find_package(catkin REQUIRED COMPONENTS roscpp std_msgs genmsg cv_bridge)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc dnn)

set(SOURCES	    src/main.cpp
		        src/ObjectDetector.hpp
		        src/ObjectDetector.cpp
		        src/Consumable.hpp)

add_message_files(FILES
		        DetectedObject.msg
		        DetectedObjects.msg)

generate_messages(DEPENDENCIES std_msgs)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

add_executable(object_detector ${SOURCES})
target_link_libraries(object_detector PUBLIC ${catkin_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(object_detector PUBLIC ./src)
add_dependencies(object_detector ${object_detector_EXPORTED_TARGETS})
