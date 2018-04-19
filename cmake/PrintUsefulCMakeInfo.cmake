# Distributed under the MIT License.
# See LICENSE.txt for details.

# Get the current working branch and commit hash
execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND git describe --abbrev=0 --always --tags
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "\nUseful Information:")
message(STATUS "Git Branch: " ${GIT_BRANCH})
message(STATUS "Git Hash: " ${GIT_COMMIT_HASH})
message(STATUS "Build Directory: " ${CMAKE_BINARY_DIR})
message(STATUS "Source Directory: " ${CMAKE_SOURCE_DIR})
message(STATUS "Bin Directory: " ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
message(STATUS "CMake Modules Path: " ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message(STATUS "CMAKE_CXX_LINK_FLAGS: " ${CMAKE_CXX_LINK_FLAGS})
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: " ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: " ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "CMAKE_EXE_LINKER_FLAGS: " ${CMAKE_EXE_LINKER_FLAGS})
message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
message(STATUS "SPECTRE_LIBRARIES: ${SPECTRE_LIBRARIES}")
message(STATUS "USE_PCH: ${USE_PCH}")
message(STATUS "USE_SYSTEM_INCLUDE: ${USE_SYSTEM_INCLUDE}")
if (PYTHONINTERP_FOUND)
  message(STATUS "Python: " ${PYTHON_EXECUTABLE})
  message(STATUS "Python Version: ${PYTHON_VERSION_STRING}")
else()
  message(STATUS "Python: Not found")
endif()
if(CLANG_TIDY_BIN)
  message(STATUS "Found clang-tidy: ${CLANG_TIDY_BIN}")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(
      STATUS
      "Could not find clang-tidy even though LLVM clang is installed"
  )
endif()

if (CODE_COVERAGE)
  message(STATUS "Code coverage enabled. All prerequisites found:")
  message(STATUS "  gcov: ${GCOV}")
  message(STATUS "  lcov: ${LCOV}")
  message(STATUS "  genhtml: ${GENHTML}")
  message(STATUS "  sed: ${SED}")
endif()

if(DOXYGEN_FOUND)
  message(STATUS "Doxygen: " ${DOXYGEN_EXECUTABLE})
else()
  message(STATUS "Doxygen: Not found, documentation cannot be built.")
endif()
