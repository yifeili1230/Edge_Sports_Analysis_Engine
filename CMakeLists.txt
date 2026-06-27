cmake_minimum_required(VERSION 3.16)
project(video_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(video_engine src/main.cpp)
target_link_libraries(video_engine PRIVATE ${OpenCV_LIBS})

enable_testing()
add_executable(motion_detector_test tests/test_motion.cpp)
target_link_libraries(motion_detector_test PRIVATE ${OpenCV_LIBS})
target_include_directories(motion_detector_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_test(NAME motion_detector_test COMMAND motion_detector_test)

add_executable(object_tracker_test tests/test_tracker.cpp)
target_link_libraries(object_tracker_test PRIVATE ${OpenCV_LIBS})
target_include_directories(object_tracker_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_test(NAME object_tracker_test COMMAND object_tracker_test)

add_executable(skeleton_renderer_test tests/test_skeleton_renderer.cpp)
target_link_libraries(skeleton_renderer_test PRIVATE ${OpenCV_LIBS})
target_include_directories(skeleton_renderer_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_test(NAME skeleton_renderer_test COMMAND skeleton_renderer_test)

add_executable(frame_timeline_test tests/test_frame_timeline.cpp)
target_link_libraries(frame_timeline_test PRIVATE ${OpenCV_LIBS})
target_include_directories(frame_timeline_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_test(NAME frame_timeline_test COMMAND frame_timeline_test)

add_executable(pose_schema_test tests/test_pose_schema.cpp)
target_link_libraries(pose_schema_test PRIVATE ${OpenCV_LIBS})
target_include_directories(pose_schema_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_test(NAME pose_schema_test COMMAND pose_schema_test)
