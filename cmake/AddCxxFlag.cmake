# Distributed under the MIT License.
# See LICENSE.txt for details.

# Checks if a CXX flag is supported by the compiler and adds it if it is
function(check_and_add_cxx_flag FLAG_TO_CHECK)
  include(CheckCXXCompilerFlag)
  unset(CXX_FLAG_WORKS CACHE)
  # In order to check for a -Wno-* flag in gcc, you have to check the
  # -W* version instead.  See http://gcc.gnu.org/wiki/FAQ#wnowarning
  string(REGEX REPLACE ^-Wno- -W FLAG_TO_CHECK_POSITIVE ${FLAG_TO_CHECK})
  set(CMAKE_REQUIRED_QUIET 1)
  check_cxx_compiler_flag(${FLAG_TO_CHECK_POSITIVE} CXX_FLAG_WORKS)
  unset(CMAKE_REQUIRED_QUIET)
  if (CXX_FLAG_WORKS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG_TO_CHECK}" PARENT_SCOPE)
  endif()
endfunction()

# Checks if a flag is supported by the linker and adds it if it is
function(check_and_add_cxx_link_flag FLAG_TO_CHECK)
  include(CheckCxxLinkerFlag)
  unset(CXX_LINKER_FLAG_WORKS CACHE)
  set(CMAKE_REQUIRED_QUIET 1)
  check_cxx_linker_flag(${FLAG_TO_CHECK} CXX_LINKER_FLAG_WORKS)
  unset(CMAKE_REQUIRED_QUIET)
  if(CXX_LINKER_FLAG_WORKS)
    set(CMAKE_CXX_LINK_FLAGS
      "${CMAKE_CXX_LINK_FLAGS} ${FLAG_TO_CHECK}" PARENT_SCOPE)
  endif()
endfunction()
