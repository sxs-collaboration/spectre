# Distributed under the MIT License.
# See LICENSE.txt for details.

option(USE_LD
  "Override the default linker. Options are: ld, gold, lld"
  OFF)

if (USE_LD)
  if("${USE_LD}" STREQUAL "gold")
    find_program(GNU_GOLD_LINKER "ld.gold")
    if (NOT GNU_GOLD_LINKER)
      message(FATAL_ERROR
        "ld.gold requested but could not find executable")
    endif()
    check_and_add_cxx_link_flag("-fuse-ld=gold")
  elseif("${USE_LD}" STREQUAL "lld")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      message(FATAL_ERROR "GCC does not support linking with LLD. "
        "If this has changed please remove this error message "
        "and submit a pull request.")
    endif()
    find_program(LLD_LINKER "ld.lld")
    if (NOT LLD_LINKER)
      message(FATAL_ERROR
        "ld.lld requested but could not find executable")
    endif()
    check_and_add_cxx_link_flag("-fuse-ld=lld")
  elseif(NOT "${USE_LD}" STREQUAL "ld")
    message(FATAL_ERROR
      "USE_LD must be one of 'ld', 'gold' or 'lld' but got '${USE_LD}'")
  endif()
else()
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # GCC currently only supports linking with ld.gold, not ld.lld
    find_program(GNU_GOLD_LINKER "ld.gold")
    if (GNU_GOLD_LINKER)
      check_and_add_cxx_link_flag("-fuse-ld=gold")
    endif()
  else()
    find_program(LLD_LINKER "ld.lld")
    if (LLD_LINKER)
      check_and_add_cxx_link_flag("-fuse-ld=lld")
    else()
      find_program(GNU_GOLD_LINKER "ld.gold")
      if (GNU_GOLD_LINKER)
        check_and_add_cxx_link_flag("-fuse-ld=gold")
      endif()
    endif()
  endif()
endif()
