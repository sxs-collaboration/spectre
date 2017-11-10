// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/TagsDeclarations.hpp"

#pragma once

namespace GeneralizedHarmonic {
template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  static constexpr bool is_euclidean = false;

  using variables_tags = typelist<SpacetimeMetric<Dim>, Pi<Dim>, Phi<Dim>>;
  using gradient_tags = variables_tags;

  using Variables = ::Variables<variables_tags>;
};
} // namespace GeneralizedHarmonic
