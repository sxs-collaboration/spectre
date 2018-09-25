// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/GeneralizedHarmonic/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"

/// \cond
namespace brigand {
template <class...>
struct list;
}  // namespace brigand

template <class>
class Variables;
/// \endcond

/*!
 * \ingroup EvolutionSystemsGroup
 * \brief Items related to evolving the first-order generalized harmonic system.
 */
namespace GeneralizedHarmonic {
template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  static constexpr bool is_euclidean = false;

  using variables_tags =
      brigand::list<gr::Tags::SpacetimeMetric<Dim>, Pi<Dim>, Phi<Dim>>;
  using gradient_tags = variables_tags;

  using Variables = ::Variables<variables_tags>;
};
}  // namespace GeneralizedHarmonic
