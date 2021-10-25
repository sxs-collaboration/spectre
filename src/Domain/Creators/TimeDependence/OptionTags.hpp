// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Options/Options.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/PrettyType.hpp"

namespace domain {
namespace creators {
namespace time_dependence {
/*!
 * \brief OptionTags for specific TimeDependences
 *
 * These are used to create Composition TimeDependences. E.g.
 * CompositionName:
 *   TimeDepOptionTag1:
 *     options...
 *   TimeDepOptionTag2:
 *     optons...
 */
namespace OptionTags {
/*!
 * OptionTag for all TimeDependences to be used in some kind of composition
 *
 * The \p Index template parameter is to keep track if more than one of the same
 * TimeDependence is used by adding the index to the end of the name. The
 * default defined in TimeDependence.hpp is 0. If the composition is of two
 * unique TimeDependences, then the index is ignored.
 */
template <typename TimeDep, size_t Index>
struct TimeDependenceCompositionTag {
  static constexpr size_t mesh_dim = TimeDep::mesh_dim;
  static std::string name() {
    std::string suffix = Index == 0 ? "" : get_output(Index);
    return pretty_type::short_name<TimeDep>() + suffix;
  }
  using type = TimeDep;
  static constexpr Options::String help = {
      "One of the maps in the composition."};
};
}  // namespace OptionTags
}  // namespace time_dependence
}  // namespace creators
}  // namespace domain
