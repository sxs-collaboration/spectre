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
  static std::string name() noexcept { return "Evolution"; }
  static constexpr OptionString help{"Options for the time evolution"};
};

/*!
 * \ingroup OptionGroupsGroup
 * \brief Groups option tags related to the evolution system.
 *
 * The option tags for the evolution system should be placed in a subgroup that
 * carries the system name. See e.g. `OptionTags::ValenciaDivCleanGroup`.
 */
struct SystemGroup {
  static std::string name() noexcept { return "EvolutionSystem"; }
  static constexpr OptionString help{"The system of hyperbolic PDEs"};
};

/*!
 * \brief Holds option tags for importing numerical initial data for an
 * evolution.
 *
 */
struct NumericalInitialData {
  using group = importers::OptionTags::Group;
  static constexpr OptionString help{"Initial data for evolution."};
};
}  // namespace OptionTags
}  // namespace evolution
