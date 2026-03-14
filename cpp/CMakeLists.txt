cmake_minimum_required(VERSION 3.16)
project(arvs_cpp VERSION 1.0.0 LANGUAGES CXX)

# ── C++ standard — C++17 minimum for std::optional, structured bindings ──────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Safety-critical compiler flags ───────────────────────────────────────────
# -fstack-protector-strong  : canary against stack overflows
# -Wall -Wextra -Wpedantic  : enable all warnings
# -Werror                   : treat warnings as errors (DO-178C style)
# -fno-exceptions           : no C++ exception overhead in hot path
# -fno-rtti                 : no RTTI overhead
# These can be relaxed during early development if needed.
add_compile_options(
    -Wall -Wextra -Wpedantic
    -fstack-protector-strong
    -fno-exceptions
    -fno-rtti
)

# Treat warnings as errors in CI / release builds
if(DEFINED ENV{ARVS_STRICT_BUILD})
    add_compile_options(-Werror)
endif()

# ── ROS2 dependencies ─────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp      REQUIRED)
find_package(std_msgs    REQUIRED)
find_package(std_srvs    REQUIRED)
find_package(arvs_msgs   REQUIRED)   # Custom message package — build this first

# ── Core library (no ROS2 dependency — portable to RTOS / bare-metal) ────────
add_library(arvs_core STATIC
    src/safety_gate.cpp
    src/axiom_validator.cpp
    src/watchdog_timer.cpp
    src/execution_controller.cpp
)

target_include_directories(arvs_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# ── ROS2 nodes ────────────────────────────────────────────────────────────────

# Safety Gate Node
add_executable(safety_gate_node
    ros2/safety_gate_node.cpp
)
target_link_libraries(safety_gate_node arvs_core)
ament_target_dependencies(safety_gate_node
    rclcpp std_msgs std_srvs arvs_msgs)

# Axiom Validator Node
add_executable(axiom_validator_node
    ros2/axiom_validator_node.cpp
)
target_link_libraries(axiom_validator_node arvs_core)
ament_target_dependencies(axiom_validator_node
    rclcpp std_msgs std_srvs arvs_msgs)

# Execution Controller Node
add_executable(execution_controller_node
    ros2/execution_controller_node.cpp
)
target_link_libraries(execution_controller_node arvs_core)
ament_target_dependencies(execution_controller_node
    rclcpp std_msgs std_srvs arvs_msgs)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS
    arvs_core
    safety_gate_node
    axiom_validator_node
    execution_controller_node
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
    DESTINATION include/${PROJECT_NAME}
)

# ── Testing ───────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)

    ament_add_gtest(test_safety_gate
        tests/test_safety_gate.cpp
    )
    target_link_libraries(test_safety_gate arvs_core)

    ament_add_gtest(test_axiom_validator
        tests/test_axiom_validator.cpp
    )
    target_link_libraries(test_axiom_validator arvs_core)

    ament_add_gtest(test_watchdog
        tests/test_watchdog_timer.cpp
    )
    target_link_libraries(test_watchdog arvs_core)
endif()

ament_package()
