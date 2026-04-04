cmake_minimum_required(VERSION 3.16)
project(QuantPricer
    VERSION 1.0.0
    DESCRIPTION "Multi-Model Derivatives Pricing Engine in C++"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Performance flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address -fno-omit-frame-pointer")

# ============================================================================
# Core Library
# ============================================================================
file(GLOB_RECURSE LIB_SOURCES "src/*.cpp")
list(FILTER LIB_SOURCES EXCLUDE REGEX "paper_trader\\.cpp$")
add_library(quantpricer STATIC ${LIB_SOURCES})
target_include_directories(quantpricer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(quantpricer PRIVATE -Wall -Wextra -Wpedantic)

# ============================================================================
# Main Demo Executable
# ============================================================================
add_executable(quantpricer_demo examples/main_demo.cpp)
target_link_libraries(quantpricer_demo PRIVATE quantpricer)

# ============================================================================
# Individual Example Executables
# ============================================================================
add_executable(mc_european examples/mc_european.cpp)
target_link_libraries(mc_european PRIVATE quantpricer)

add_executable(mc_exotic examples/mc_exotic.cpp)
target_link_libraries(mc_exotic PRIVATE quantpricer)

add_executable(fdm_solver examples/fdm_solver.cpp)
target_link_libraries(fdm_solver PRIVATE quantpricer)

add_executable(greeks_engine examples/greeks_engine.cpp)
target_link_libraries(greeks_engine PRIVATE quantpricer)

add_executable(vol_surface examples/vol_surface.cpp)
target_link_libraries(vol_surface PRIVATE quantpricer)

add_executable(heston_pricer examples/heston_pricer.cpp)
target_link_libraries(heston_pricer PRIVATE quantpricer)

add_executable(jump_diffusion examples/jump_diffusion.cpp)
target_link_libraries(jump_diffusion PRIVATE quantpricer)

add_executable(american_options examples/american_options.cpp)
target_link_libraries(american_options PRIVATE quantpricer)

add_executable(barrier_options examples/barrier_options.cpp)
target_link_libraries(barrier_options PRIVATE quantpricer)

add_executable(multi_asset_options examples/multi_asset_options.cpp)
target_link_libraries(multi_asset_options PRIVATE quantpricer)

add_executable(risk_management examples/risk_management.cpp)
target_link_libraries(risk_management PRIVATE quantpricer)

add_executable(fixed_income examples/fixed_income.cpp)
target_link_libraries(fixed_income PRIVATE quantpricer)

add_executable(rate_models examples/rate_models.cpp)
target_link_libraries(rate_models PRIVATE quantpricer)

add_executable(orderbook_demo examples/orderbook_demo.cpp)
target_link_libraries(orderbook_demo PRIVATE quantpricer)

# ============================================================================
# Paper Trading Bot (Self-Simulated — no network deps)
# ============================================================================

# nlohmann/json — prefer system install, fallback to FetchContent
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3)
    FetchContent_MakeAvailable(json)
endif()

add_executable(paper_trader src/paper_trader.cpp)
target_link_libraries(paper_trader PRIVATE
    quantpricer
    nlohmann_json::nlohmann_json)
target_include_directories(paper_trader PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ============================================================================
# Sample Data Generator (Day 2)
# ============================================================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_sample_data.cpp")
    add_executable(generate_sample_data tools/generate_sample_data.cpp)
    target_link_libraries(generate_sample_data PRIVATE quantpricer)
    target_include_directories(generate_sample_data PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

# ============================================================================
# Backtest Runner (Day 13)
# ============================================================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/run_backtest.cpp")
    add_executable(run_backtest tools/run_backtest.cpp)
    target_link_libraries(run_backtest PRIVATE quantpricer nlohmann_json::nlohmann_json)
    target_include_directories(run_backtest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

# ============================================================================
# Test Suite
# ============================================================================
enable_testing()
add_executable(test_runner tests/test_runner.cpp)
target_link_libraries(test_runner PRIVATE quantpricer)
add_test(NAME QuantPricerTests COMMAND test_runner)

message(STATUS "QuantPricer v${PROJECT_VERSION} configured successfully")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
