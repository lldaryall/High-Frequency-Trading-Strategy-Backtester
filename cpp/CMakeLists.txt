cmake_minimum_required(VERSION 3.12)
project(flashback_cpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
set(Python_EXECUTABLE "/Users/daryapylypenko/.pyenv/shims/python")
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Use the specific Python executable
set(PYTHON_EXECUTABLE "/Users/daryapylypenko/.pyenv/shims/python")
set(Python_EXECUTABLE "/Users/daryapylypenko/.pyenv/shims/python")

# Get pybind11 path from Python
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Find pybind11
set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED)

# Set output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../flashback/market)

# Create the pybind11 module
pybind11_add_module(_match_cpp
    pybind_module.cpp
    match_engine.cpp
)

# Set target properties
set_target_properties(_match_cpp PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    OUTPUT_NAME "_match_cpp"
    SUFFIX ".so"
)

# Compiler-specific options
if(MSVC)
    target_compile_options(_match_cpp PRIVATE /W4)
else()
    target_compile_options(_match_cpp PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native)
endif()

# Link libraries
target_link_libraries(_match_cpp PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)

# Install the module
install(TARGETS _match_cpp
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../flashback/market
    RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../flashback/market
)

# Print configuration info
message(STATUS "Building Flashback C++ matching engine")
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "pybind11 version: ${pybind11_VERSION}")
message(STATUS "Output directory: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")

# Optional: Add custom target for building with specific Python version
if(DEFINED PYTHON_EXECUTABLE)
    set_target_properties(_match_cpp PROPERTIES
        PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE}
    )
endif()
