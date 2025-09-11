// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/SnakeCase.hpp"

#include <string>

std::string camel_case_to_snake_case(const std::string& input) {
  std::string output;
  for (const char c : input) {
    if (static_cast<bool>(std::isupper(c)) and not output.empty()) {
      output += '_';
    }
    output += static_cast<char>(std::tolower(c));
  }
  return output;
}

std::string snake_case_to_camel_case(const std::string& input) {
  std::string output;
  bool to_upper = true;
  for (const char c : input) {
    if (c == '_') {
      to_upper = true;
    } else {
      output += to_upper ? static_cast<char>(std::toupper(c)) : c;
      to_upper = false;
    }
  }
  return output;
}
