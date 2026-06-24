// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <concepts>
#include <cstddef>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/ConstructibleFromTags.hpp"
#include "Utilities/Serialization/Serializable.hpp"

/// \cond
template <size_t VolumeDim>
class ElementId;
/// \endcond

namespace evolution::dg {
/// Concept for a class usable in `EqualRateRegions`.
template <typename T, size_t Dim>
concept equal_rate_region_generator =
    std::default_initializable<T> and std::movable<T> and serializable<T> and
    db::constructible_from_tags<T> and
    requires(const T gen, const size_t region_id,
             const ElementId<Dim> element_id) {
      {
        gen.regions()
      } -> std::same_as<std::unordered_map<std::string, size_t>>;
      { gen.is_in_region(region_id, element_id) } -> std::same_as<bool>;
    };
}  // namespace evolution::dg
