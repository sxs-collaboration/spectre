// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/Formaline.hpp"

namespace formaline {
std::vector<char> get_archive() {
  return {'N', 'o', 't', ' ', 's', 'u', 'p', 'p', 'o', 'r', 't', 'e', 'd'};
}

std::string get_environment_variables() {
  return "Not supported in Python";
}

std::string get_build_info() {
  return "Not supported in Python";
}

std::string get_paths() { return "Not supported in Python."; }
}  // namespace formaline
