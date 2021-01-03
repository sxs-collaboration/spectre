# Distributed under the MIT License.
# See LICENSE.txt for details.

message(STATUS "\nUseful Information:")
message(STATUS "Git Branch: " ${GIT_BRANCH})
message(STATUS "Git Description: " ${GIT_DESCRIPTION})
message(STATUS "Git Hash: " ${GIT_HASH})
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
message(STATUS "BUILD_SHARED_LIBS: " ${BUILD_SHARED_LIBS})
message(STATUS "USE_PCH: ${USE_PCH}")
message(STATUS "ASAN: ${ASAN}")
message(STATUS "UBSAN_UNDEFINED: ${UBSAN_UNDEFINED}")
message(STATUS "UBSAN_INTEGER: ${UBSAN_INTEGER}")
message(STATUS "USE_SYSTEM_INCLUDE: ${USE_SYSTEM_INCLUDE}")
if (Python_FOUND)
  message(STATUS "Python: " ${Python_EXECUTABLE})
  message(STATUS "Python Version: ${Python_VERSION}")
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
