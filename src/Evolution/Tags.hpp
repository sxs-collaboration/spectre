// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "Options/Options.hpp"

namespace OptionTags {

/*!
 * \ingroup OptionGroupsGroup
 * \brief Groups option tags related to the time evolution, e.g. time step and
 * time stepper.
 */
struct EvolutionGroup {
  static std::string name() noexcept { return "Evolution"; }
  static constexpr OptionString help{"Options for the time evolution"};
};

}  // namespace OptionTags
