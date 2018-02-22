// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "@CMAKE_SOURCE_DIR@/src/Informer/InfoFromBuild.hpp"

#include <boost/preprocessor.hpp>
#include <sstream>

namespace {
std::string link_date() { return std::string(__TIMESTAMP__); }

std::string git_commit_hash() {
  return std::string(BOOST_PP_STRINGIZE(GIT_COMMIT_HASH));
}

std::string git_branch() { return std::string(BOOST_PP_STRINGIZE(GIT_BRANCH)); }
}  // namespace

std::string spectre_version() { return std::string("@SpECTRE_VERSION@"); }

int spectre_major_version() { return @SpECTRE_VERSION_MAJOR@; }

int spectre_minor_version() { return @SpECTRE_VERSION_MINOR@; }

int spectre_patch_version() { return @SpECTRE_VERSION_PATCH@; }

std::string info_from_build() {
  static const std::string info = [] {
    std::ostringstream os;
    os << "SpECTRE Build Information:\n";
    os << "Version:                      " << spectre_version() << "\n";
    os << "Compiled on host:             @HOSTNAME@\n";
    os << "Compiled in directory:        @CMAKE_BINARY_DIR@\n";
    os << "Source directory is:          @CMAKE_SOURCE_DIR@\n";
    os << "Compiled on git branch:       " << git_branch() << "\n";
    os << "Compiled with git hash:       " << git_commit_hash() << "\n";
    os << "Linked on:                    " << link_date() << "\n";
    return os.str();
  }();
  return info;
}

std::string unit_test_path() noexcept {
  return "@CMAKE_SOURCE_DIR@/tests/Unit/";
}
