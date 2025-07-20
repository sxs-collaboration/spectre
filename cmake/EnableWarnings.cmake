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
-Wwrite-strings" SpectreWarnings)
else()
  add_library(SpectreWarnings INTERFACE)
endif()

# Disable some warnings
create_cxx_flags_target(
    "-Wno-dangling-reference;\
-Wno-documentation-unknown-command;\
-Wno-mismatched-tags;\
-Wno-interference-size;\
-Wno-non-template-friend;\
-Wno-type-limits;\
-Wno-undefined-var-template;\
-Wno-gnu-zero-variadic-macro-arguments;\
-Wno-noexcept-type"
  SpectreDisableSomeWarnings)
target_link_libraries(
  SpectreWarnings
  INTERFACE
  SpectreDisableSomeWarnings
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

# Suppress CUDA warnings that we don't want
create_cxx_flag_target(
  "-Xcudafe \"--diag_suppress=177,186,191,554,1301,1305,2189,3060,20012\""
  SpectreCudaWarnings)
target_link_libraries(
  SpectreWarnings
  INTERFACE
  SpectreCudaWarnings
  )

# Have Clang warn about unsupported CUDA versions but don't error
create_cxx_flags_target(
  "-Wno-error=unknown-cuda-version"
  SpectreCudaClangWarnings)
target_link_libraries(
  SpectreWarnings
  INTERFACE
  SpectreCudaClangWarnings
  )

target_link_libraries(
  SpectreFlags
  INTERFACE
  SpectreWarnings
  )
