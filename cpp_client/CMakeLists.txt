cmake_minimum_required(VERSION 3.10)
project(tf_cpp_client)

# Use C++14 standard
set(CMAKE_CXX_STANDARD 14)

# Include directories for header files
include_directories(${CMAKE_SOURCE_DIR}/include)

# Link directories for external libraries
link_directories(${CMAKE_SOURCE_DIR}/lib)

# Add your source files here
add_executable(main
    main.cpp
    TextPreprocessor.cpp
    LabelUtils.cpp
)

# Link TensorFlow library
target_link_libraries(main tensorflow)

# After build, copy the TensorFlow DLL to the output directory (for Windows runtime)
add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/lib/tensorflow.dll"
        $<TARGET_FILE_DIR:main>
)