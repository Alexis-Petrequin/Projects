cmake_minimum_required(VERSION 3.0)


#############################
#   BASE
#############################


# Set C++ standard
set(CMAKE_CXX_STANDARD              14)
set(CMAKE_CXX_STANDARD_REQUIRED     ON)

# Set include and link directories
include_directories(libs)
link_directories(libs)


#############################
#   BOMBERMAN
#############################


# Create project
project(bomberman)

# Get project sources and headers files
file(GLOB SRCS
    "src/*.hpp"
    "src/*.cpp")

# Add binary executable
add_executable(${PROJECT_NAME} ${SRCS})

# Copy ressources
add_custom_command(
    TARGET ${PROJECT_NAME} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/ressources/ $<TARGET_FILE_DIR:${PROJECT_NAME}>/ressources/)


#############################
#   PLATFORMS
#############################


IF (WIN32)

    # Set subsystem type
    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE)

    # Set linked libraries
    target_link_libraries(${PROJECT_NAME} raylib winmm)

ELSE()

    # Set linked libraries
    target_link_libraries(${PROJECT_NAME} raylib pthread dl)

ENDIF()
