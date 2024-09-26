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
-Wcast-align;\
-Wcast-qual;\
-Wdisabled-optimization;\
-Wdocumentation;\
-Wextra;\
-Wformat-nonliteral;\
-Wformat-security;\
-Wformat-y2k;\
-Wformat=2;\
-Winvalid-pch;\
-Wmissing-declarations;\
-Wmissing-field-initializers;\
-Wmissing-format-attribute;\
-Wmissing-include-dirs;\
-Wmissing-noreturn;\
-Wnewline-eof;\
-Wno-dangling-reference;\
-Wno-documentation-unknown-command;\
-Wno-mismatched-tags;\
-Wno-non-template-friend;\
-Wno-type-limits;\
-Wno-undefined-var-template;\
-Wnon-virtual-dtor;\
-Wold-style-cast;\
-Woverloaded-virtual;\
-Wpacked;\
-Wpedantic;\
-Wpointer-arith;\
-Wredundant-decls;\
-Wshadow;\
-Wsign-conversion;\
-Wstack-protector;\
-Wswitch-default;\
-Wunreachable-code;\
-Wno-gnu-zero-variadic-macro-arguments;\
-Wwrite-strings" SpectreWarnings)
endif()

# GCC 7.1ish and newer warn about noexcept changing mangled names,
# but we don't care
create_cxx_flag_target("-Wno-noexcept-type" SpectreWarnNoNoexceptType)

target_link_libraries(
  SpectreWarnings
  INTERFACE
  SpectreWarnNoNoexceptType
  )

# GCC versions below 13 don't respect 'GCC diagnostic' pragmas to disable
# warnings by the preprocessor:
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=53431
# So we disable the warning about unknown pragmas because we can't silence it.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
    AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13)
  create_cxx_flag_target("-Wno-unknown-pragmas" SpectreWarnNoUnknownPragmas)
  target_link_libraries(
    SpectreWarnings
    INTERFACE
    SpectreWarnNoUnknownPragmas
    )
endif()

target_link_libraries(
  SpectreFlags
  INTERFACE
  SpectreWarnings
  )
