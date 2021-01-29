// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivative.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
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
      gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>,
      Tags::Pi<Dim, Frame::Inertial>, Tags::Phi<Dim, Frame::Inertial>>>;
  using flux_variables = tmpl::list<>;
  using gradient_variables =
      tmpl::list<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>,
                 Tags::Pi<Dim, Frame::Inertial>,
                 Tags::Phi<Dim, Frame::Inertial>>;
  using gradients_tags = gradient_variables;

  using compute_volume_time_derivative_terms = TimeDerivative<Dim>;
  using normal_dot_fluxes = ComputeNormalDotFluxes<Dim>;

  using char_speeds_compute_tag =
      CharacteristicSpeedsCompute<Dim, Frame::Inertial>;
  using char_speeds_tag = Tags::CharacteristicSpeeds<Dim, Frame::Inertial>;
  using compute_largest_characteristic_speed =
      ComputeLargestCharacteristicSpeed<Dim, Frame::Inertial>;

  template <typename Tag>
  using magnitude_tag = ::Tags::NonEuclideanMagnitude<
      Tag, gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>>;

  using inverse_spatial_metric_tag =
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>;
};
}  // namespace GeneralizedHarmonic
