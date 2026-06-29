cmake_minimum_required(VERSION 3.16)
project(video_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn)

add_executable(video_engine src/main.cpp)
target_include_directories(video_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(video_engine PRIVATE ${OpenCV_LIBS})

option(VIDEO_ENGINE_ENABLE_TENSORRT "Enable native TensorRT inference when available" ON)
if(VIDEO_ENGINE_ENABLE_TENSORRT)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        PATH_SUFFIXES aarch64-linux-gnu x86_64-linux-gnu)
    find_path(CUDART_INCLUDE_DIR cuda_runtime_api.h
        PATHS /usr/local/cuda
        PATH_SUFFIXES include targets/aarch64-linux/include targets/x86_64-linux/include)
    find_library(TENSORRT_LIBRARY nvinfer)
    find_library(CUDART_LIBRARY cudart
        PATHS /usr/local/cuda/lib64
        /usr/local/cuda/targets/aarch64-linux/lib
        /usr/local/cuda/targets/x86_64-linux/lib)
endif()

if(TENSORRT_INCLUDE_DIR AND CUDART_INCLUDE_DIR AND TENSORRT_LIBRARY AND CUDART_LIBRARY)
    target_compile_definitions(video_engine PRIVATE VIDEO_ENGINE_WITH_TENSORRT=1)
    target_include_directories(video_engine PRIVATE
        ${TENSORRT_INCLUDE_DIR}
        ${CUDART_INCLUDE_DIR})
    target_link_libraries(video_engine PRIVATE ${TENSORRT_LIBRARY} ${CUDART_LIBRARY})
    message(STATUS "Native TensorRT pose inference: enabled")
else()
    target_compile_definitions(video_engine PRIVATE VIDEO_ENGINE_WITH_TENSORRT=0)
    message(STATUS "Native TensorRT pose inference: disabled")
endif()

add_executable(video_engine_diagnostics src/diagnostics.cpp)
target_link_libraries(video_engine_diagnostics PRIVATE ${OpenCV_LIBS})

include(CTest)
if(BUILD_TESTING)
    function(add_video_engine_test test_name source_name)
        add_executable(${test_name}_test tests/test_${source_name}.cpp)
        target_link_libraries(${test_name}_test PRIVATE ${OpenCV_LIBS})
        target_include_directories(${test_name}_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
        add_test(NAME ${test_name}_test COMMAND ${test_name}_test)
    endfunction()

    add_video_engine_test(skeleton_renderer skeleton_renderer)
    add_video_engine_test(frame_timeline frame_timeline)
    add_video_engine_test(pose_schema pose_schema)
    add_video_engine_test(resize_processor resize_processor)
    add_video_engine_test(yolo_pose_decoder yolo_pose_decoder)
    add_video_engine_test(squat_analyzer squat_analyzer)
    add_video_engine_test(ohp_analyzer ohp_analyzer)
endif()
