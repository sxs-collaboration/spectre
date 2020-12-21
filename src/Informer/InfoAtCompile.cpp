// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "@CMAKE_SOURCE_DIR@/src/Informer/InfoFromBuild.hpp"

std::string spectre_version() { return std::string("@SPECTRE_VERSION@"); }

std::string unit_test_path() noexcept {
  return "@CMAKE_SOURCE_DIR@/tests/Unit/";
}
