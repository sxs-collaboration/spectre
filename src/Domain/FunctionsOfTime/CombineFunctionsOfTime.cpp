// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FunctionsOfTime/CombineFunctionsOfTime.hpp"

#include <string>

namespace domain::FunctionsOfTime {
FoTMap combine_functions_of_time(const FoTMap& map_1, const FoTMap& map_2) {
  FoTMap result{};
  for (auto& [name, f_of_t] : map_1) {
    result[name] = f_of_t->get_clone();
  }
  for (auto& [name, f_of_t] : map_2) {
    result[name] = f_of_t->get_clone();
  }
  return result;
}
}  // namespace domain::FunctionsOfTime
