# Distributed under the MIT License.
# See LICENSE.txt for details.

include(AddCxxFlag)

# On systems where we can't use -isystem (Cray), we don't want
# all the warnings enabled because we get flooded with system warnings.
option(ENABLE_WARNINGS "Enable the default warning level" ON)
if(${ENABLE_WARNINGS})
  check_and_add_cxx_flag("-W")
  check_and_add_cxx_flag("-Wall")
  check_and_add_cxx_flag("-Wextra")
  check_and_add_cxx_flag("-Wpedantic")

  check_and_add_cxx_flag("-Wcast-align")
  check_and_add_cxx_flag("-Wcast-qual")
  check_and_add_cxx_flag("-Wdisabled-optimization")
  check_and_add_cxx_flag("-Wdocumentation")
  check_and_add_cxx_flag("-Wformat=2")
  check_and_add_cxx_flag("-Wformat-nonliteral")
  check_and_add_cxx_flag("-Wformat-security")
  check_and_add_cxx_flag("-Wformat-y2k")
  check_and_add_cxx_flag("-Winvalid-pch")
  check_and_add_cxx_flag("-Wmissing-field-initializers")
  check_and_add_cxx_flag("-Wmissing-format-attribute")
  check_and_add_cxx_flag("-Wmissing-include-dirs")
  check_and_add_cxx_flag("-Wmissing-noreturn")
  check_and_add_cxx_flag("-Wnewline-eof")
  check_and_add_cxx_flag("-Wno-documentation-unknown-command")
  check_and_add_cxx_flag("-Wno-mismatched-tags")
  check_and_add_cxx_flag("-Wnon-virtual-dtor")
  check_and_add_cxx_flag("-Wold-style-cast")
  check_and_add_cxx_flag("-Woverloaded-virtual")
  check_and_add_cxx_flag("-Wpacked")
  check_and_add_cxx_flag("-Wpointer-arith")
  check_and_add_cxx_flag("-Wredundant-decls")
  check_and_add_cxx_flag("-Wshadow")
  check_and_add_cxx_flag("-Wsign-conversion")
  check_and_add_cxx_flag("-Wstack-protector")
  check_and_add_cxx_flag("-Wswitch-default")
  check_and_add_cxx_flag("-Wunreachable-code")
  check_and_add_cxx_flag("-Wwrite-strings")
endif()

# GCC 7.1ish and newer warn about noexcept changing mangled names,
# but we don't care
check_and_add_cxx_flag("-Wno-noexcept-type")

check_and_add_cxx_link_flag("-Qunused-arguments")
