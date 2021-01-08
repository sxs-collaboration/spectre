// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <charm++.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <pup.h>
#include <stdexcept>
#include <string>
#include <vector>

// In order to not get runtime errors that CmiPrintf, etc. are not defined we
// provide resonable replacements for them. Getting the correct Charm++ library
// linked is non-trivial since Charm++ generally wants you to use their `charmc`
// script, but we want to avoid that.

/// \cond

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

// Identification on system
#ifndef CmiMyPe
int CmiMyPe() { return 0; }
#endif
int _Cmi_mype = 0;
int _Cmi_numpes = 1;
int _Cmi_mynodesize = 1;
int _Cmi_mynode = 0;
int _Cmi_numnodes = 1;

// CmiWallTimer support
double _cpu_speed_factor = 0.0;
double CmiTimer(void) { return 0.0; }
#ifndef CmiWallTimer
double CmiWallTimer(void) { return 0.0; }
#endif

// We need to maintain Charm-compatibility, which means not marking things with
// attributes.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-noreturn"
// Charm LRTS locks. These should never be called from the python code so we
// throw an exception if they are.
LrtsNodeLock LrtsCreateLock() {
  throw std::runtime_error{
      "The function 'LrtsCreateLock' is provided for compatibility with "
      "Charm++ and is not intended to be called."};
  // clang-tidy wants us to use nullptr but I'm worried that LrtsNodeLock isn't
  // always a pointer
  return 0;  // NOLINT
}
void LrtsLock(LrtsNodeLock /*lock*/) {
  throw std::runtime_error{
      "The function 'LrtsLock' is provided for compatibility with "
      "Charm++ and is not intended to be called."};
}
void LrtsUnlock(LrtsNodeLock /*lock*/) {
  throw std::runtime_error{
      "The function 'LrtsUnlock' is provided for compatibility with "
      "Charm++ and is not intended to be called."};
}
int LrtsTryLock(LrtsNodeLock /*lock*/) {
  throw std::runtime_error{
      "The function 'LrtsTryLock' is provided for compatibility with "
      "Charm++ and is not intended to be called."};
}
void LrtsDestroyLock(LrtsNodeLock /*lock*/) {
  throw std::runtime_error{
      "The function 'LrtsDestroyLock' is provided for compatibility with "
      "Charm++ and is not intended to be called."};
}

void CmiAbort(const char* msg) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
  fprintf(stderr, "%s", msg);
  abort();
}

void realCkExit(int exitcode) { exit(exitcode); }
#pragma GCC diagnostic pop

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

/*
 * PUP::able support
 *
 * Here we provide dummy-implementations of functions in Charm++'s `pup.h` so
 * that we can write Python wrappers for classes that derive from `PUP::able`.
 * These functions are not intended to be called, so we throw exceptions that
 * propagate to Python and provide an error message when one of the functions
 * gets called accidentally.
 */

// Destructor will be called in Python, but doesn't need to do anything
PUP::able::~able() = default;

// gcc and clang suggest to mark these functions `noreturn`, but they are
// declared in pup.h
#pragma GCC diagnostic push
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif  // defined(__GNUC__) && !defined(__clang__)
#ifdef __clang__
#pragma GCC diagnostic ignored "-Wmissing-noreturn"
#endif  // __clang__

void PUP::able::pup(PUP::er& /*p*/) {
  throw std::runtime_error{
      "The function 'PUP::able::pup' is provided for compatibility with "
      "Charm++ and is not intended to be called."};
}

PUP::able::PUP_ID PUP::able::register_constructor(const char* /*className*/,
                                                  constructor_function /*fn*/) {
  throw std::runtime_error{
      "The function 'PUP::able::register_constructor' is provided for "
      "compatibility with Charm++ and is not intended to be called."};
}

PUP::able* PUP::able::clone() const {
  throw std::runtime_error{
      "The function 'PUP::able::clone' is provided for compatibility with "
      "Charm++ and is not intended to be called."};
}

void pup_bytes(pup_er /*p*/, void* /*ptr*/, size_t /*nBytes*/) {
  throw std::runtime_error{
      "The function 'pup_bytes' is provided for compatibility with "
      "Charm++ and is not intended to be called."};
}

#pragma GCC diagnostic pop

/// \endcond
