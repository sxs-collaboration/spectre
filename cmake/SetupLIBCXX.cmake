# Distributed under the MIT License.
# See LICENSE.txt for details.



if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  if (NOT USE_LIBCXX)
    option(USE_LIBCXX "Use libc++ as standard library" OFF)
  endif()
  if (USE_LIBCXX)
    include(CheckCXXCompilerFlag)
    unset(CXX_FLAG_WORKS CACHE)
    set(CMAKE_REQUIRED_QUIET 1)
    check_cxx_compiler_flag("-stdlib=libc++" CXX_FLAG_WORKS)
    unset(CMAKE_REQUIRED_QUIET)
    # If we cannot use the -stdlib=libc++ flag directly we must find libc++
    # explicity by searching on the file system
    if (CXX_FLAG_WORKS AND NOT LIBCXX_ROOT)
      message(STATUS "libc++: Compiler supports -stdlib=libc++, no need to find "
        "libc++ explicity. Specify `LIBCXX_ROOT` to use a specified version of "
        "libc++. If the specified version is not found we fall back to the "
        "system libc++, if available.")
    else()
      find_package(LIBCXX REQUIRED)
      set(
        CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}\
 -L${LIBCXXABI_LIBRARIES} -L${LIBCXX_LIBRARIES}"
        )
      include_directories(${LIBCXX_INCLUDE_DIRS})
      message(STATUS "libc++ include: ${LIBCXX_INCLUDE_DIRS}")
      message(STATUS "libc++ libraries: ${LIBCXX_LIBRARIES}")
    endif()

    set(CMAKE_CXX_FLAGS "-stdlib=libc++ ${CMAKE_CXX_FLAGS}")

    option(
      LIBCXX_DEBUG
      "Enable debug mode for libc++ in Debug builds"
      OFF
      )
    if(LIBCXX_DEBUG)
      set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -D_LIBCPP_DEBUG=0")
    endif()
  endif()
endif()
