// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/String.hpp"
#include "Utilities/PrettyType.hpp"

namespace CurvedScalarWave {
namespace OptionTags {

struct BackgroundSpacetimeGroup {
  static constexpr Options::String help = {
      "The background spacetime on which the scalar wave "
      "propagates."};
  static std::string name() { return "BackgroundSpacetime"; }
};

template <typename BackgroundType>
struct BackgroundSpacetime {
  static constexpr Options::String help = {
      "Options for the background spacetime on which the scalar wave "
      "propagates."};
  static std::string name() {
    return pretty_type::short_name<BackgroundType>();
  }
  using type = BackgroundType;
  using group = BackgroundSpacetimeGroup;
};
}  // namespace OptionTags

namespace Tags {
/*!
 * \brief The background spacetime on which the scalar wave propagates.
 */
template <typename BackgroundType>
struct BackgroundSpacetime : db::SimpleTag {
  using type = BackgroundType;
  using option_tags =
      tmpl::list<OptionTags::BackgroundSpacetime<BackgroundType>>;
  static constexpr bool pass_metavariables = false;
  static BackgroundType create_from_options(const BackgroundType& background) {
    return background;
  }
};
}  // namespace Tags
}  // namespace CurvedScalarWave
