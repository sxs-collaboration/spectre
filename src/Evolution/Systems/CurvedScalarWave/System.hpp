// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"
#include "Evolution/Systems/CurvedScalarWave/Equations.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/TimeDerivative.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
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

  using boundary_correction_base = BoundaryCorrections::BoundaryCorrection<Dim>;

  using variables_tag =
      ::Tags::Variables<tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>>>;
  using flux_variables = tmpl::list<>;
  using gradient_variables = tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>>;

  // Relic alias: needs to be removed once all evolution systems
  // convert to using dg::ComputeTimeDerivative
  using gradients_tags = gradient_variables;

  using compute_volume_time_derivative_terms = TimeDerivative<Dim>;
  using normal_dot_fluxes = ComputeNormalDotFluxes<Dim>;

  using compute_largest_characteristic_speed =
      Tags::ComputeLargestCharacteristicSpeed<Dim>;

  using inverse_spatial_metric_tag =
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>;
};
}  // namespace CurvedScalarWave
