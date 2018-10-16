// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// In order to not get runtime errors that CmiPrintf, etc. are not defined we
// provide resonable replacements for them. Getting the correct Charm++ library
// linked is non-trivial since Charm++ generally wants you to use their `charmc`
// script, but we want to avoid that.

__attribute__ ((format (printf, 1, 2)))
// NOLINTNEXTLINE(cert-dcl50-cpp)
void CmiPrintf(const char* fmt, ...) {
  va_list args;
  // clang-tidy: cppcoreguidelines-pro-type-vararg,
  // cppcoreguidelines-pro-bounds-array-to-pointer-decay
  va_start(args, fmt);  // NOLINT
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
  vprintf(fmt, args);
  // clang-tidy: cppcoreguidelines-pro-type-vararg,
  // cppcoreguidelines-pro-bounds-array-to-pointer-decay
  va_end(args);  // NOLINT
}

__attribute__ ((format (printf, 1, 2)))
// NOLINTNEXTLINE(cert-dcl50-cpp)
void CmiError(const char* fmt, ...) {
  va_list args;
  // clang-tidy: cppcoreguidelines-pro-type-vararg,
  // cppcoreguidelines-pro-bounds-array-to-pointer-decay
  va_start(args, fmt);  // NOLINT
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
  vfprintf(stderr, fmt, args);
  // clang-tidy: cppcoreguidelines-pro-type-vararg,
  // cppcoreguidelines-pro-bounds-array-to-pointer-decay
  va_end(args);  // NOLINT
}

int CmiMyPe() { return 0; }
int _Cmi_mynode = 0;
std::string info_from_build() { return "build_info"; }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-noreturn"
void CmiAbort(const char* msg) {
#pragma GCC diagnostic pop
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
  fprintf(stderr, "%s", msg);
  abort();
}
namespace formaline {
std::vector<char> get_archive() noexcept {
  return {'N', 'o', 't', ' ', 's', 'u', 'p', 'p', 'o', 'r', 't', 'e', 'd'};
}

std::string get_environment_variables() noexcept {
  return "Not supported on macOS";
}

std::string get_library_versions() noexcept {
  return "Not supported in python";
}

std::string get_paths() noexcept { return "Not supported in python."; }
}  // namespace formaline
