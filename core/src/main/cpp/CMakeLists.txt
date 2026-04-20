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

# Add linker flag for 16 KB page alignment
set_target_properties(hnswlib_jni PROPERTIES
        LINK_FLAGS "-Wl,-z,max-page-size=16384"
)

target_link_libraries(
        hnswlib_jni
        ${log-lib}
)