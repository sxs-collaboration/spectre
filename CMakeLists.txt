
# Distributed under the MIT License.
# See LICENSE.txt for details.

cmake_minimum_required(VERSION 3.3.2)

project(SpECTRE VERSION 0.0.0 LANGUAGES CXX C)

set(CMAKE_VERBOSE_MAKEFILE OFF)

site_name(HOSTNAME)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(CheckCompilerVersion)
include(ProhibitInSourceBuild)
include(SetOutputDirectory)
include(SetBuildType)
include(SetCxxStandard)
include(EnableWarnings)
include(SetupBuildInfo)
include(SetupCharm)
include(SetupCharmProjections)

# In order to use certain code analysis tools like clang-tidy and cppcheck the
# compile commands need to be accessible. CMake can write these to a
# "compile_commands.json" file.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(SetupLibCXX)
include(SetupClangFormat)
include(SetupClangTidy)
include(SetupCppCheck)
include(SetupSanitizers)
include(SetupListTargets)
include(SetupDoxygen)

include(SetupBlas)
include(SetupBoost)
include(SetupCatch)
include(SetupHdf5)
include(SetupPapi)
include(SetupYamlCpp)

enable_testing(true)
include(SpectreAddCatchTests)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR})

# Charm++ generated headers are created in the build directory
include_directories(SYSTEM ${CMAKE_BINARY_DIR})
include_directories(SYSTEM ${CMAKE_BINARY_DIR}/src)

add_subdirectory(src)
add_subdirectory(tests)

include(PrintUsefulCMakeInfo)
