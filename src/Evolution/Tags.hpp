// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "IO/Importers/Tags.hpp"
#include "Options/Options.hpp"

namespace evolution {
namespace OptionTags {

/*!
 * \ingroup OptionGroupsGroup
 * \brief Groups option tags related to the time evolution, e.g. time step and
 * time stepper.
 */
struct Group {
  static std::string name() { return "Evolution"; }
  static constexpr Options::String help{"Options for the time evolution"};
};

/*!
 * \ingroup OptionGroupsGroup
 * \brief Groups option tags related to the evolution system.
 *
 * The option tags for the evolution system should be placed in a subgroup that
 * carries the system name. See e.g. `OptionTags::ValenciaDivCleanGroup`.
 */
struct SystemGroup {
  static std::string name() { return "EvolutionSystem"; }
  static constexpr Options::String help{"The system of hyperbolic PDEs"};
};

/*!
 * \ingroup OptionGroupsGroup
 * \brief Holds option tags for importing numeric initial data for an evolution.
 */
struct NumericInitialData {
  using group = importers::OptionTags::Group;
  static constexpr Options::String help =
      "Numeric initial data for all system variables";
};

}  // namespace OptionTags
}  // namespace evolution
