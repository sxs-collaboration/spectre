# Distributed under the MIT License.
# See LICENSE.txt for details.

# Checks if a CXX flag is supported by the compiler and adds it if it is
function(check_and_add_cxx_flag FLAG_TO_CHECK)
  include(CheckCXXCompilerFlag)
  unset(CXX_FLAG_WORKS CACHE)
  set(CMAKE_REQUIRED_QUIET 1)
  check_cxx_compiler_flag(${FLAG_TO_CHECK} CXX_FLAG_WORKS)
  unset(CMAKE_REQUIRED_QUIET)
  if (CXX_FLAG_WORKS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG_TO_CHECK}" PARENT_SCOPE)
  endif()
endfunction()
