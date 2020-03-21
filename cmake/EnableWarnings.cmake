# Distributed under the MIT License.
# See LICENSE.txt for details.

include(AddCxxFlag)

# On systems where we can't use -isystem (Cray), we don't want
# all the warnings enabled because we get flooded with system warnings.
option(ENABLE_WARNINGS "Enable the default warning level" ON)
if(${ENABLE_WARNINGS})
  create_cxx_flags_target(
    "-W;\
-Wall;\
-Wextra;\
-Wpedantic;\
-Wcast-align;\
-Wcast-qual;\
-Wdisabled-optimization;\
-Wdocumentation;\
-Wformat=2;\
-Wformat-nonliteral;\
-Wformat-security;\
-Wformat-y2k;\
-Winvalid-pch;\
-Wmissing-field-initializers;\
-Wmissing-format-attribute;\
-Wmissing-include-dirs;\
-Wmissing-noreturn;\
-Wnewline-eof;\
-Wno-documentation-unknown-command;\
-Wno-mismatched-tags;\
-Wnon-virtual-dtor;\
-Wold-style-cast;\
-Woverloaded-virtual;\
-Wpacked;\
-Wpointer-arith;\
-Wredundant-decls;\
-Wshadow;\
-Wsign-conversion;\
-Wstack-protector;\
-Wswitch-default;\
-Wunreachable-code;\
-Wwrite-strings" SpectreWarnings)
endif()

# GCC 7.1ish and newer warn about noexcept changing mangled names,
# but we don't care
create_cxx_flag_target("-Wno-noexcept-type" SpectreWarnNoNoexceptType)

check_and_add_cxx_link_flag("-Qunused-arguments")

target_link_libraries(
  SpectreWarnings
  INTERFACE
  SpectreWarnNoNoexceptType
  )

target_link_libraries(
  SpectreFlags
  INTERFACE
  SpectreWarnings
  )
