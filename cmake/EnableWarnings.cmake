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

# - GCC versions below 13 don't respect 'GCC diagnostic' pragmas to disable
#   warnings by the preprocessor:
#   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=53431
#   So we disable the warning about unknown pragmas because we can't silence it.
# - GCC has many false positives for `stringop-overflow`, `array-bounds`, and
#   `restrict`, specifically in libstdc++ <string> with C++20, leading to
#   warnings from `__builtin_memcpy`.
# - GCC has false-positive `use-after-free` warnings from Blaze's DynamicMatrix
#   constructor that gets inlined in many places. Rather than silencing the
#   warning at every call site or forcing no-inline, we silence it here.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  create_cxx_flags_target(
    "-Wno-unknown-pragmas;\
-Wno-stringop-overflow;\
-Wno-stringop-overread;\
-Wno-maybe-uninitialized;\
-Wno-array-bounds;\
-Wno-restrict;\
-Wno-use-after-free"
    SpectreDisableGccWarnings)
  target_link_libraries(
    SpectreWarnings
    INTERFACE
    SpectreDisableGccWarnings
    )
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
    AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 22
    AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 23)
  create_cxx_flags_target(
    "-Wno-c2y-extensions"
    SpectreDisableClangCatchWarning)
  target_link_libraries(
    SpectreWarnings
    INTERFACE
    SpectreDisableClangCatchWarning
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

target_link_libraries(
  SpectreFlags
  INTERFACE
  SpectreWarnings
  )
