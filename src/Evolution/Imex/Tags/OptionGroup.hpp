// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "Evolution/Tags.hpp"
#include "Options/String.hpp"

namespace imex::OptionTags {
/// Option group for IMEX options
struct Group {
  static std::string name() { return "Imex"; }
  static constexpr Options::String help{"Options for IMEX evolution"};
  using group = evolution::OptionTags::Group;
};
}  // namespace imex::OptionTags
