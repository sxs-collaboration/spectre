# Distributed under the MIT License.
# See LICENSE.txt for details.

cmake_minimum_required(VERSION 3.12.0)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Determine the project version and other metadata
include(SpectreLoadMetadata)

# Set up the project. Notes:
# - Fortran is needed for SPHEREPACK
project(${SPECTRE_NAME} VERSION ${SPECTRE_VERSION} LANGUAGES CXX C Fortran)

# Unset the CMake-defined version variable because it strips zeros from the
# version components, e.g. 2020.12.07 becomes 2020.12.7
unset(${SPECTRE_NAME}_VERSION)
# Also unset the version-component variables because they have no meaning in
# our versioning scheme
unset(${SPECTRE_NAME}_VERSION_MAJOR)
unset(${SPECTRE_NAME}_VERSION_MINOR)
unset(${SPECTRE_NAME}_VERSION_PATCH)

# Policies
# The `cmake_minimum_required` above sets policies to `NEW` that are compatible
# with the given minimum cmake version. Here we overwrite policies that we
# have back-ported in our cmake code.
# - We use `<PACKAGE_NAME>_ROOT` variables to provide search paths for
#   dependencies
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.12)
  cmake_policy(SET CMP0074 NEW)
endif ()

set(CMAKE_VERBOSE_MAKEFILE OFF)

site_name(HOSTNAME)

include(SpectreInitializeVariables)
include(SpectreGetGitHash)

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
include(SpectreSetupFlagsTarget)
include(SetupNinjaColors)
include(SetOutputDirectory)
include(SpectreAddInterfaceLibraryHeaders)
include(SpectreTargetHeaders)
include(SpectreTargetSources)
include(SetupFormaline)
# We need yapf before we set up the git hooks
include(SetupYapf)
include(SetupGitHooks)
include(SetBuildType)
include(SetupPic)
include(SetCxxStandard)
include(StripSymbols)
# We need Boost for InfoAtLink
include(SetupBoost)
include(SetupBuildInfo)
include(SetupCCache)
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
include(SetupProfiling)
include(SetupSanitizers)
include(SetupListTargets)
include(AddSpectreExecutable)
include(CheckBrokenArray0)

# Blaze depends on both Blas and Lapack, since LIBXSMM is a Blas
# alternative we set it up early too.
include(SetupBlas)
include(SetupLapack)
include(SetupLIBXSMM)

include(SetupBlaze)
include(SetupBrigand)
include(SetupCatch)
include(SetupGoogleBenchmark)
include(SetupGsl)
include(SetupHdf5)
include(SetupAllocator)
include(SetupPapi)
include(SetupPybind11)
include(SetupStl)
include(SetupLibsharp)
include(SetupYamlCpp)

include(SetupLIBCXXCharm)
include(SpectreSetupPythonPackage)
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
include(SetupSphinx)
include(CodeCoverageDetection)
include(SpectreAddLibraries)

enable_testing(true)
include(SetupPypp)
include(SpectreAddTestLibs)
include(SpectreAddCatchTests)
include(AddInputFileTests)

include_directories(${CMAKE_SOURCE_DIR}/external)
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/tests/Unit)

# Charm++ generated headers are created in the build directory
spectre_include_directories(${CMAKE_BINARY_DIR})
spectre_include_directories(${CMAKE_BINARY_DIR}/src)
spectre_include_directories(${CMAKE_BINARY_DIR}/src/Parallel)

add_subdirectory(external)
add_subdirectory(src)
add_subdirectory(tests)

include(PrintUsefulCMakeInfo)
include(SetupVersionFile)

include(SpectreCheckDependencies)
