// Distributed under the MIT License.
// See LICENSE.txt for details.

// This file is configured into the build directory by CMake and compiled as
// part of the build system. It exposes information from CMake to C++.

#include "@CMAKE_SOURCE_DIR@/src/Informer/InfoFromBuild.hpp"

#include <sstream>
#include <string>

std::string spectre_version() { return std::string("@SPECTRE_VERSION@"); }

std::string unit_test_build_path() noexcept {
  return "@CMAKE_BINARY_DIR@/tests/Unit/";
}

std::string unit_test_src_path() noexcept {
  return "@CMAKE_SOURCE_DIR@/tests/Unit/";
}

std::string info_from_build() noexcept {
  std::ostringstream os;
  os << "SpECTRE Build Information:\n";
  os << "Version:                      " << spectre_version() << "\n";
  os << "Compiled on host:             @HOSTNAME@\n";
  os << "Compiled in directory:        @CMAKE_BINARY_DIR@\n";
  os << "Source directory is:          @CMAKE_SOURCE_DIR@\n";
  os << "Compiled on git branch:       " << git_branch() << "\n";
  os << "Compiled on git revision:     " << git_description() << "\n";
  os << "Linked on:                    " << link_date() << "\n";
  return os.str();
}
