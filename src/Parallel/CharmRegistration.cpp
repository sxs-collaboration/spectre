// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/CharmRegistration.hpp"

#include <algorithm>
#include <cstddef>
#include <string>

namespace Parallel::charmxx {
std::string get_template_parameters_as_string_impl(
    const std::string& function_name) {
  std::string template_params =
      function_name.substr(function_name.find(std::string("Args = ")) + 8);
  template_params.erase(template_params.end() - 2, template_params.end());
  size_t pos = 0;
  while ((pos = template_params.find(" >")) != std::string::npos) {
    template_params.replace(pos, 1, ">");
    template_params.erase(pos + 1, 1);
  }
  pos = 0;
  while ((pos = template_params.find(", ", pos)) != std::string::npos) {
    template_params.erase(pos + 1, 1);
  }
  pos = 0;
  while ((pos = template_params.find('>', pos + 2)) != std::string::npos) {
    template_params.replace(pos, 1, " >");
  }
  std::replace(template_params.begin(), template_params.end(), '%', '>');
  // GCC's __PRETTY_FUNCTION__ adds the return type at the end, so we remove it.
  if (template_params.find('}') != std::string::npos) {
    template_params.erase(template_params.find('}'), template_params.size());
  }
  // Remove all spaces
  const auto new_end =
      std::remove(template_params.begin(), template_params.end(), ' ');
  template_params.erase(new_end, template_params.end());
  return template_params;
}
}  // namespace Parallel::charmxx
