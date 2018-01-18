# Distributed under the MIT License.
# See LICENSE.txt for details.

cmake_minimum_required(VERSION 3.3.2)

# Fortran below is needed for SPHEREPACK.
project(SpECTRE VERSION 0.0.0 LANGUAGES CXX C Fortran)

set(CMAKE_VERBOSE_MAKEFILE OFF)

site_name(HOSTNAME)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(CheckCompilerVersion)
include(ProhibitInSourceBuild)
include(SetOutputDirectory)
include(SetupGitHooks)
include(SetBuildType)
include(SetCxxStandard)
include(SetupBuildInfo)
include(SetupCCache)
include(SetupVersionFile)
include(SetupCharm)
include(SetupCharmProjections)
include(SetupMacOsx)
include(EnableWarnings)
include(SetupGoldOrLldLinker)

# In order to use certain code analysis tools like clang-tidy and cppcheck the
# compile commands need to be accessible. CMake can write these to a
# "compile_commands.json" file.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(SetupLIBCXX)
include(SetupClangFormat)
include(SetupClangTidy)
include(SetupCppCheck)
include(SetupCxxFlags)
include(SetupSanitizers)
include(SetupListTargets)
include(SetupDoxygen)
include(CodeCoverageDetection)

include(SetupBlas)
include(SetupBlaze)
include(SetupBoost)
include(SetupBrigand)
include(SetupCatch)
include(SetupGoogleBenchmark)
include(SetupGsl)
include(SetupHdf5)
include(SetupJemalloc)
include(SetupLIBXSMM)
include(SetupPapi)
include(SetupYamlCpp)

# The precompiled header must be setup after all libraries have been found
include(SetupPch)

# Generate Charm++ files
generate_algorithms(${CMAKE_BINARY_DIR}/src/Parallel/Algorithms)

enable_testing(true)
include(SpectreAddCatchTests)
include(SetupCompilationFailureTests)

include_directories(${CMAKE_SOURCE_DIR}/external)
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR})

# Charm++ generated headers are created in the build directory
include_directories(SYSTEM ${CMAKE_BINARY_DIR})
include_directories(SYSTEM ${CMAKE_BINARY_DIR}/src)
include_directories(SYSTEM ${CMAKE_BINARY_DIR}/src/Parallel)

add_subdirectory(external)
add_subdirectory(src)
add_subdirectory(tests)

include(PrintUsefulCMakeInfo)
