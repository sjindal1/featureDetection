cmake_minimum_required(VERSION 3.2)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(APP_NAME feature_detection)
project(${APP_NAME})

set (CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )


add_executable(${APP_NAME} app.cpp)

target_link_libraries(${APP_NAME} ${OpenCV_LIBS})

message("${OpenCV_LIBS}")
