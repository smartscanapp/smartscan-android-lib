cmake_minimum_required(VERSION 3.10)
project(hnswlib_jni)

add_library(
        hnswlib_jni
        SHARED
        hnswlib_jni.cpp
)

include_directories(
        ${CMAKE_SOURCE_DIR}/hnswlib
)

find_library(
        log-lib
        log
)

target_link_libraries(
        hnswlib_jni
        ${log-lib}
)
