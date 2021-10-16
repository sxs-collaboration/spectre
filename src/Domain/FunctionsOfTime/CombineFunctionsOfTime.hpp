// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <unordered_map>

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"

namespace domain::FunctionsOfTime {
using FoTMap = std::unordered_map<std::string, std::unique_ptr<FunctionOfTime>>;
FoTMap combine_functions_of_time(const FoTMap& map_1, const FoTMap& map_2);
}  // namespace domain::FunctionsOfTime
