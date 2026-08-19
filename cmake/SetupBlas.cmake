# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(BLAS REQUIRED)
message(STATUS "BLAS libs: " ${BLAS_LIBRARIES})
file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "BLAS_LIBRARIES: ${BLAS_LIBRARIES}\n"
  )

set(IS_OPENBLAS FALSE)

if(BLAS_FOUND)
  string(TOLOWER "${BLAS_LIBRARIES}" BLAS_LIBS_LOWER)
  if(BLAS_LIBS_LOWER MATCHES "openblas")
    set(IS_OPENBLAS TRUE)
  else()
    if(TARGET BLAS::BLAS)
      get_target_property(BLAS_LINK_LIBS BLAS::BLAS INTERFACE_LINK_LIBRARIES)
      string(TOLOWER "${BLAS_LINK_LIBS}" BLAS_TARGET_LIBS_LOWER)
      if(BLAS_TARGET_LIBS_LOWER MATCHES "openblas")
        set(IS_OPENBLAS TRUE)
      endif()
    endif()
  endif()
endif()

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  BLAS::BLAS
  )

# Check if we have found OpenBLAS and can disable its multithreading, since it
# conflicts with Charm++ parallelism. Details:
# https://github.com/xianyi/OpenBLAS/wiki/Faq#multi-threaded
try_compile(CHECK_DISABLE_OPENBLAS_MULTITHREADING_RESULT
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/cmake/CheckOpenBlasThreads.cpp
  LINK_LIBRARIES BLAS::BLAS
  OUTPUT_VARIABLE DISABLE_CHECK_OUTPUT
)

if(CHECK_DISABLE_OPENBLAS_MULTITHREADING_RESULT)
  set(DISABLE_OPENBLAS_MULTITHREADING ON)
  add_compile_definitions(DISABLE_OPENBLAS_MULTITHREADING)
  message(STATUS "Disabled OpenBLAS multithreading")
else()
  if(IS_OPENBLAS)
    message(STATUS ${DISABLE_CHECK_OUTPUT})
    message(FATAL_ERROR
            "BLAS vendor is OpenBLAS but disabling multithreading failed.")
  else()
    message(STATUS "BLAS vendor is not OpenBLAS. Make sure it doesn't "
      "try to do multithreading that might conflict with Charm++ parallelism.")
  endif()
endif()
