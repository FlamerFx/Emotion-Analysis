cmake_minimum_required(VERSION 3.10)
project(tf_cpp_client)

set(CMAKE_CXX_STANDARD 14)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/imgui)
# Add GLFW include if needed:
# include_directories(${CMAKE_SOURCE_DIR}/third_party/glfw/include)

link_directories(${CMAKE_SOURCE_DIR})
link_directories(${CMAKE_SOURCE_DIR}/lib)
# link_directories(${CMAKE_SOURCE_DIR}/third_party/glfw/lib-vc2022) # If using local GLFW

add_executable(gui_main
    main.cpp
    TextPreprocessor.cpp
    LabelUtils.cpp
    imgui.cpp
    imgui_draw.cpp
    imgui_tables.cpp
    imgui_widgets.cpp
    imgui_impl_glfw.cpp
    imgui_impl_opengl3.cpp
)

# Link with TensorFlow, GLFW, and OpenGL
target_link_libraries(gui_main tensorflow glfw3 opengl32)

# Optional: Copy DLL to build dir
add_custom_command(TARGET gui_main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/lib/tensorflow.dll"
        $<TARGET_FILE_DIR:gui_main>
)