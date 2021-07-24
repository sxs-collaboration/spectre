# Distributed under the MIT License.
# See LICENSE.txt for details.

# First append useful info to BuildInfo.txt
file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "CMake version: ${CMAKE_VERSION}\n"
  "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}\n"
  "CMAKE_CXX_LINK_FLAGS: ${CMAKE_CXX_LINK_FLAGS}\n"
  "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}\n"
  "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}\n"
  "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}\n"
  "CMAKE_C_FLAGS_DEBUG: ${CMAKE_C_FLAGS_DEBUG}\n"
  "CMAKE_C_FLAGS_RELEASE: ${CMAKE_C_FLAGS_RELEASE}\n"
  "CMAKE_Fortran_FLAGS: ${CMAKE_Fortran_FLAGS}\n"
  "CMAKE_Fortran_FLAGS_DEBUG: ${CMAKE_Fortran_FLAGS_DEBUG}\n"
  "CMAKE_Fortran_FLAGS_RELEASE: ${CMAKE_Fortran_FLAGS_RELEASE}\n"
  "CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}\n"
  "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}\n"
  "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}\n"
  "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}\n"
  "CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}\n"
  "CMAKE_C_COMPILER_VERSION: ${CMAKE_C_COMPILER_VERSION}\n"
  "CMAKE_Fortran_COMPILER: ${CMAKE_Fortran_COMPILER}\n"
  "CMAKE_Fortran_COMPILER_VERSION: ${CMAKE_Fortran_COMPILER_VERSION}\n"
  "Python version: ${Python_VERSION}\n"
  )

# Then write (slightly expanded) useful info to command line
message(STATUS "\nUseful Information:")
message(STATUS "Git description: " ${GIT_DESCRIPTION})
message(STATUS "Git branch: " ${GIT_BRANCH})
message(STATUS "Git hash: " ${GIT_HASH})
message(STATUS "Build directory: " ${CMAKE_BINARY_DIR})
message(STATUS "Source directory: " ${CMAKE_SOURCE_DIR})
message(STATUS "Bin directory: " ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
message(STATUS "CMake modules path: " ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message(STATUS "CMAKE_CXX_LINK_FLAGS: " ${CMAKE_CXX_LINK_FLAGS})
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: " ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: " ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "CMAKE_EXE_LINKER_FLAGS: " ${CMAKE_EXE_LINKER_FLAGS})
message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
message(STATUS "BUILD_SHARED_LIBS: " ${BUILD_SHARED_LIBS})
message(STATUS "USE_PCH: ${USE_PCH}")
message(STATUS "ASAN: ${ASAN}")
message(STATUS "UBSAN_UNDEFINED: ${UBSAN_UNDEFINED}")
message(STATUS "UBSAN_INTEGER: ${UBSAN_INTEGER}")
message(STATUS "USE_SYSTEM_INCLUDE: ${USE_SYSTEM_INCLUDE}")

if (Python_FOUND)
  message(STATUS "Python: " ${Python_EXECUTABLE})
  message(STATUS "Python version: ${Python_VERSION}")
else()
  message(STATUS "Python: Not found")
endif()
message(STATUS "BUILD_PYTHON_BINDINGS: " ${BUILD_PYTHON_BINDINGS})

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

if(BUILD_PYTHON_BINDINGS AND "${JEMALLOC_LIB_TYPE}" STREQUAL SHARED)
  message(STATUS
    "When using the python bindings you must run python as:\n"
    "   LD_PRELOAD=${JEMALLOC_LIBRARIES} python ...\n"
    "Alternatively, use the system allocator by setting \n"
    "-D MEMORY_ALLOCATOR=SYSTEM")
endif()
