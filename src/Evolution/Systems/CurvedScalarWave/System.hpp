// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"
#include "Evolution/Systems/CurvedScalarWave/Equations.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/TimeDerivative.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \ingroup EvolutionSystemsGroup
 * \brief Items related to evolving a scalar wave on a curved background
 */
namespace CurvedScalarWave {

template <size_t Dim>
struct System {
  static constexpr bool is_in_flux_conservative_form = false;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = Dim;
  static constexpr bool is_euclidean = false;

  using variables_tag = ::Tags::Variables<tmpl::list<Pi, Phi<Dim>, Psi>>;
  using flux_variables = tmpl::list<>;
  using gradient_variables = tmpl::list<Pi, Phi<Dim>, Psi>;

  // Relic alias: needs to be removed once all evolution systems
  // convert to using dg::ComputeTimeDerivative
  using gradients_tags = gradient_variables;

  using spacetime_variables_tag = ::Tags::Variables<tmpl::list<
      gr::Tags::Lapse<DataVector>, ::Tags::dt<gr::Tags::Lapse<DataVector>>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>,
                    Frame::Inertial>,
      gr::Tags::Shift<volume_dim, Frame::Inertial, DataVector>,
      ::Tags::dt<gr::Tags::Shift<volume_dim, Frame::Inertial, DataVector>>,
      ::Tags::deriv<gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
                    tmpl::size_t<Dim>, Frame::Inertial>,
      gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataVector>,
      ::Tags::dt<
          gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataVector>>,
      ::Tags::deriv<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataVector>,
                    tmpl::size_t<Dim>, Frame::Inertial>,
      gr::Tags::SqrtDetSpatialMetric<DataVector>,
      gr::Tags::ExtrinsicCurvature<volume_dim, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<volume_dim, Frame::Inertial, DataVector>>>;

  using compute_volume_time_derivative_terms = TimeDerivative<Dim>;
  using normal_dot_fluxes = ComputeNormalDotFluxes<Dim>;

  using char_speeds_compute_tag = CharacteristicSpeedsCompute<Dim>;
  using char_speeds_tag = Tags::CharacteristicSpeeds<Dim>;
  using compute_largest_characteristic_speed =
      ComputeLargestCharacteristicSpeed<Dim>;

  template <typename Tag>
  using magnitude_tag = ::Tags::NonEuclideanMagnitude<
      Tag, gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>>;
};
}  // namespace CurvedScalarWave
