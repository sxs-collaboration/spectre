// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/Scalar/Events/ObserveSelfForce.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/AnalyticData/CircularOrbit.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace ScalarSelfForce::Events::detail {

std::optional<tnsr::i<std::complex<double>, 2>> extract_self_force(
    const Domain<2>& domain, const ElementId<2>& element_id,
    const AnalyticData::CircularOrbit& circular_orbit,
    const Scalar<ComplexDataVector>& field, const Mesh<2>& mesh,
    const InverseJacobian<DataVector, 2, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian) {
  // Get element-logical coords of puncture
  const auto puncture_position = circular_orbit.puncture_position();
  const auto& block = domain.blocks()[element_id.block_id()];
  const auto block_logical_coords =
      block_logical_coordinates_single_point(puncture_position, block);
  if (not block_logical_coords.has_value()) {
    return std::nullopt;
  }
  const auto puncture_logical_coords =
      element_logical_coordinates(block_logical_coords.value(), element_id);
  if (not puncture_logical_coords.has_value()) {
    return std::nullopt;
  }
  // Interpolate field and its derivatives to puncture location
  const auto deriv_field = partial_derivative(field, mesh, inv_jacobian);
  const intrp::Irregular<2> interpolator(mesh, puncture_logical_coords.value());
  ComplexDataVector intrp_result{1_st};
  Scalar<std::complex<double>> field_at_puncture{};
  interpolator.interpolate(make_not_null(&intrp_result), get(field));
  get(field_at_puncture) = intrp_result[0];
  tnsr::i<std::complex<double>, 2> deriv_field_at_puncture{};
  interpolator.interpolate(make_not_null(&intrp_result), get<0>(deriv_field));
  get<0>(deriv_field_at_puncture) = intrp_result[0];
  // Assuming equatorial symmetry, so theta derivative must be zero
  // (violated only by truncation error)
  get<1>(deriv_field_at_puncture) = 0.;
  // Calculate self-force in r and theta coordinates
  tnsr::i<std::complex<double>, 2> self_force = deriv_field_at_puncture;
  const double r0 = circular_orbit.orbital_radius();
  const double M = circular_orbit.black_hole_mass();
  const double spin = circular_orbit.black_hole_spin();
  const double a = M * spin;
  const int m_mode = circular_orbit.m_mode_number();
  const double r_plus = M * (1. + sqrt(1. - square(spin)));
  const double r_minus = M * (1. - sqrt(1. - square(spin)));
  get<0>(self_force) /= r0;
  if (not circular_orbit.penetrating_horizon()) {
    const double alpha = 1. - 2. * M * r0 / (square(r0) + square(a));
    get<0>(self_force) /= alpha;
  }
  get<0>(self_force) -= get(field_at_puncture) / square(r0);
  if (m_mode > 0) {
    const double delta_phi =
        m_mode * a / (r_plus - r_minus) * log((r0 - r_plus) / (r0 - r_minus));
    const double delta_phi_dr = m_mode * a / (r0 - r_minus) / (r0 - r_plus);
    get<0>(self_force) +=
        std::complex<double>{0., delta_phi_dr / r0} * get(field_at_puncture);
    const std::complex<double> rotation =
        cos(delta_phi) + std::complex<double>(0., 1.) * sin(delta_phi);
    get<0>(self_force) *= 2. * rotation;
  }
  return self_force;
}

}  // namespace ScalarSelfForce::Events::detail
