// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/Stringize.hpp"

#include <ios>
#include <string>

#include "Utilities/MakeString.hpp"

std::string stringize(const bool t) {
  return MakeString{} << std::boolalpha << t;
}
