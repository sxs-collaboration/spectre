// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace Tags {
template <class>
class Variables;
}  // namespace Tags
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

  using variables_tag = ::Tags::Variables<tmpl::list<
      gr::Tags::SpacetimeMetric<Dim, Frame::Physical, DataVector>,
      Tags::Pi<Dim, Frame::Physical>, Tags::Phi<Dim, Frame::Physical>>>;
  using gradients_tags =
      tmpl::list<gr::Tags::SpacetimeMetric<Dim, Frame::Physical, DataVector>,
                 Tags::Pi<Dim, Frame::Physical>,
                 Tags::Phi<Dim, Frame::Physical>>;

  using compute_time_derivative = ComputeDuDt<Dim>;
  using normal_dot_fluxes = ComputeNormalDotFluxes<Dim>;
  using char_speeds_tag = CharacteristicSpeedsCompute<Dim, Frame::Physical>;
  using compute_largest_characteristic_speed =
      ComputeLargestCharacteristicSpeed<Dim, Frame::Physical>;

  template <typename Tag>
  using magnitude_tag = ::Tags::NonEuclideanMagnitude<
      Tag, gr::Tags::InverseSpatialMetric<Dim, Frame::Physical, DataVector>>;
};
}  // namespace GeneralizedHarmonic
