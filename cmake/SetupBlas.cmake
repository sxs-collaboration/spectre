# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(BLAS REQUIRED)
message(STATUS "BLAS libs: " ${BLAS_LIBRARIES})
file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "BLAS_LIBRARIES:  ${BLAS_LIBRARIES}\n"
  )

add_library(Blas INTERFACE IMPORTED)
set_property(TARGET Blas
  APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${BLAS_LIBRARIES})

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Blas
  )

# Check if we have found OpenBLAS and can disable its multithreading, since it
# conflicts with Charm++ parallelism. Details:
# https://github.com/xianyi/OpenBLAS/wiki/Faq#multi-threaded
# We use `execute_process` instead of `try_compile` to avoid potentially slow
# disk IO.
set(
  CHECK_DISABLE_OPENBLAS_MULTITHREADING_SOURCE
  "extern \"C\" { void openblas_set_num_threads(int); }\n\
int main() { openblas_set_num_threads(1); }"
  )
string(REPLACE ";" " " BLAS_LIBRARIES_JOINED_WITH_SPACES "${BLAS_LIBRARIES}")
execute_process(
  COMMAND
  bash -c
  "${CMAKE_CXX_COMPILER} ${BLAS_LIBRARIES_JOINED_WITH_SPACES} -x c++ - <<< $'\
${CHECK_DISABLE_OPENBLAS_MULTITHREADING_SOURCE}' -o /dev/null"
  RESULT_VARIABLE CHECK_DISABLE_OPENBLAS_MULTITHREADING_RESULT
  ERROR_VARIABLE CHECK_DISABLE_OPENBLAS_MULTITHREADING_ERROR
  OUTPUT_QUIET
  )
if(${CHECK_DISABLE_OPENBLAS_MULTITHREADING_RESULT} EQUAL 0)
  set(DISABLE_OPENBLAS_MULTITHREADING ON)
  add_definitions(-DDISABLE_OPENBLAS_MULTITHREADING)
  message(STATUS "Disabled OpenBLAS multithreading")
else()
  message(STATUS "BLAS vendor is probably not OpenBLAS. Make sure it doesn't "
    "try to do multithreading that might conflict with Charm++ parallelism.")
endif()
