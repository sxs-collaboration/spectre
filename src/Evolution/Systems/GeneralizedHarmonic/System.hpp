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
  static constexpr bool is_in_flux_conservative_form = false;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = Dim;
  static constexpr bool is_euclidean = false;

  using variables_tags = brigand::list<gr::Tags::SpacetimeMetric<Dim>,
                                       Tags::Pi<Dim>, Tags::Phi<Dim>>;
  using gradient_tags = variables_tags;

  using Variables = ::Variables<variables_tags>;
};
}  // namespace GeneralizedHarmonic
