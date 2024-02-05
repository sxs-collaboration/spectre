// Distributed under the MIT License.
// See LICENSE.txt for details.

// This file defines some symbols for compatibility with InfoAtLink and Charm++
// when the BundledExporter library is linked into an external program.

#include "Informer/InfoFromBuild.hpp"
#include "Utilities/Formaline.hpp"

std::string link_date() { return "Unavailable"; }

std::string executable_name() { return "Unavailable"; }

std::string git_description() { return "Unavailable"; }

std::string git_branch() { return "Unavailable"; }

namespace formaline {
std::vector<char> get_archive() {
  return {'N', 'o', 't', ' ', 's', 'u', 'p', 'p', 'o', 'r', 't', 'e', 'd'};
}

std::string get_environment_variables() { return "Unavailable"; }

std::string get_build_info() { return "Unavailable"; }

std::string get_paths() { return "Unavailable."; }
}  // namespace formaline

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-declarations"
extern "C" void CkRegisterMainModule() {}
#pragma GCC diagnostic pop
