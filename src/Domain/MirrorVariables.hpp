// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace Tags {

template <size_t Dim, typename SourceDirectionsTag, typename VariablesTag,
          typename InvertSignForTags>
struct MirrorVariables : VariablesTag, db::ComputeTag {
  using base = VariablesTag;
  using all_interior_vars_tag =
      ::Tags::Interface<SourceDirectionsTag, VariablesTag>;
  using argument_tags =
      tmpl::list<::Tags::Direction<Dim>, all_interior_vars_tag>;
  using volume_tags = tmpl::list<all_interior_vars_tag>;
  static auto function(
      const ::Direction<Dim>& direction,
      const db::item_type<all_interior_vars_tag>& all_interior_vars) noexcept {
    db::item_type<VariablesTag> exterior_vars{};
    // By default, mirror the variables on the interior to the exterior
    exterior_vars = all_interior_vars.at(direction);
    // For the specified variables, invert the sign of the mirrored values
    const db::item_type<VariablesTag> negative_interior_vars{
        -1. * all_interior_vars.at(direction)};
    tmpl::for_each<InvertSignForTags>(
        [&exterior_vars, &negative_interior_vars ](const auto tag_v) noexcept {
          using tag = tmpl::type_from<decltype(tag_v)>;
          get<tag>(exterior_vars) = get<tag>(negative_interior_vars);
        });
    return exterior_vars;
  }
};

}  // namespace Tags
