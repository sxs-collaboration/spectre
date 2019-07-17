// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "@CMAKE_SOURCE_DIR@/src/Informer/InfoFromBuild.hpp"

std::string spectre_version() { return std::string("@SpECTRE_VERSION@"); }

int spectre_major_version() { return @SpECTRE_VERSION_MAJOR@; }

int spectre_minor_version() { return @SpECTRE_VERSION_MINOR@; }

int spectre_patch_version() { return @SpECTRE_VERSION_PATCH@; }

std::string unit_test_path() noexcept {
  return "@CMAKE_SOURCE_DIR@/tests/Unit/";
}
