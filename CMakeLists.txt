# Distributed under the MIT License.
# See LICENSE.txt for details.

cmake_minimum_required(VERSION 3.3.2)

# Fortran below is needed for SPHEREPACK.
project(SpECTRE VERSION 0.0.0 LANGUAGES CXX C Fortran)

set(CMAKE_VERBOSE_MAKEFILE OFF)

site_name(HOSTNAME)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(
  DOCS_ONLY
  "Skip all initialization not required for rendering documentation"
)
if (DOCS_ONLY)
  include(SetupDoxygen)
  return()
endif (DOCS_ONLY)

include(CheckCompilerVersion)
include(ProhibitInSourceBuild)
include(SetupNinjaColors)
include(SetOutputDirectory)
include(SetupGitHooks)
include(SetBuildType)
include(SetupPic)
include(SetCxxStandard)
include(SetupBuildInfo)
include(SetupCCache)
include(SetupVersionFile)
include(SetupCraySupport)
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
include(SetupCxxFlags)
include(SetupSanitizers)
include(SetupListTargets)
include(AddSpectreExecutable)
include(CheckBrokenArray0)

# Need to load python libs before boost.python.
include(SetupPythonLibs)

include(SetupBlas)
include(SetupBlaze)
include(SetupBoost)
include(SetupBrigand)
include(SetupCatch)
include(SetupGoogleBenchmark)
include(SetupGsl)
include(SetupHdf5)
include(SetupAllocator)
include(SetupLIBXSMM)
include(SetupLapack)
include(SetupNumPy)
include(SetupPapi)
include(SetupSciPy)
include(SetupLibsharp)
include(SetupYamlCpp)

include(SetupLIBCXXCharm)
include(SpectreSetupPython)
# The precompiled header must be setup after all libraries have been found
include(SetupPch)

# All special targets and configs that need to be applied to *all*
# executables must be added at once in the 'UpdateAddExecutables' file.
# This is because of what is likely a bug in CMake where if a function is
# overridden multiple times (using the _function_name(...) method) then some
# versions of CMake (at least 3.13.2) segfault.
include(UpdateAddExecutables)

# The ClangFormat, clang-tidy, CppCheck, Doxygen, and CodeCov are intentionally
# after the PCH setup because that way they are able to change their
# dependencies on the PCH if necessary.
include(SetupClangFormat)
include(SetupClangTidy)
include(SetupIwyu)
include(SetupCppCheck)
include(SetupDoxygen)
include(CodeCoverageDetection)
include(SpectreAddLibraries)

# Generate Charm++ files
generate_algorithms(${CMAKE_BINARY_DIR}/src/Parallel/Algorithms)

enable_testing(true)
include(SpectreAddTestLibs)
include(SpectreAddCatchTests)
include(AddInputFileTests)
include(SetupCompilationFailureTests)

include_directories(${CMAKE_SOURCE_DIR}/external)
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR})

# Charm++ generated headers are created in the build directory
spectre_include_directories(${CMAKE_BINARY_DIR})
spectre_include_directories(${CMAKE_BINARY_DIR}/src)
spectre_include_directories(${CMAKE_BINARY_DIR}/src/Parallel)

add_subdirectory(external)
add_subdirectory(src)
add_subdirectory(tests)

# Set up stages for compiling test libraries in subsets.
include(SetupBuildStageTargets)

include(PrintUsefulCMakeInfo)
