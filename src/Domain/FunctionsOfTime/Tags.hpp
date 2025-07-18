// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"

/// \cond
namespace domain::FunctionsOfTime {
struct FunctionOfTime;
}  // namespace domain::FunctionsOfTime
/// \endcond

namespace domain::Tags {
/// Tag to retrieve the FunctionsOfTime from the GlobalCache.
struct FunctionsOfTime : db::SimpleTag {
  using type = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;
};
}  // namespace domain::Tags
