# 1. Set the minimum required CMake version
cmake_minimum_required(VERSION 3.20)

# 2. Define your project name, version, and language (CXX is C++)
project(MyProject 
        VERSION 1.0.0 
        LANGUAGES CXX)

# 3. Enforce the C++ standard (e.g., C++17 or C++20)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 4. Define the executable target and add source files
add_executable(neural_network
    src/neuralnetwork/neural_net.cpp 
    src/matvec/mat.cpp
    src/matvec/vec.cpp
)

# 5. Tell CMake where to look for your header files
target_include_directories(neural_network PRIVATE src/matvec/include)
