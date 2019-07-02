// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Options/Options.hpp"

namespace evolution {
namespace Tags {

struct InitialTime : db::SimpleTag {
  static std::string name() noexcept { return "InitialTime"; }
  using type = double;
};

struct InitialTimeDelta : db::SimpleTag {
  static std::string name() noexcept { return "InitialTimeDelta"; }
  using type = double;
};

struct InitialSlabSize : db::SimpleTag {
  static std::string name() noexcept { return "InitialSlabSize"; }
  using type = double;
};

}  // namespace Tags

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

/*!
 * \ingroup OptionGroupsGroup
 * \brief Groups option tags related to the evolution system.
 *
 * The option tags for the evolution system should be placed in a subgroup that
 * carries the system name. See e.g. `OptionTags::ValenciaDivCleanGroup`.
 */
struct EvolutionSystemGroup {
  static std::string name() noexcept { return "EvolutionSystem"; }
  static constexpr OptionString help{"The system of hyperbolic PDEs"};
};

}  // namespace OptionTags
}  // namespace evolution
