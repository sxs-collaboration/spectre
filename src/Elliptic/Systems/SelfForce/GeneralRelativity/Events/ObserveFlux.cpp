// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/Events/ObserveFlux.hpp"

#include <cmath>
#include <complex>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbit.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace GrSelfForce::Events::detail {

std::pair<double, double> extract_flux(
    const AnalyticData::CircularOrbit& circular_orbit,
    const tnsr::aa<ComplexDataVector, 3>& field, const Mesh<2>& mesh,
    const Scalar<DataVector>& face_jacobian,
    const tnsr::I<DataVector, 2, Frame::Inertial>& face_coords) {
  const int m_mode = circular_orbit.m_mode_number();
  const double omega = circular_orbit.omega();
  const auto field_on_face =
      dg::project_tensor_to_boundary(field, mesh, Direction<2>::upper_xi());
  const bool penetrating_horizon = circular_orbit.penetrating_horizon();
  DataVector sin_theta;
  const size_t num_face_pts = get<0>(face_coords).size();
  DataVector integrand_multiplier{num_face_pts};
  if (penetrating_horizon) {
    integrand_multiplier = 1.0;
  } else {
    sin_theta = sin(get<1>(face_coords));
    integrand_multiplier = sin_theta;
  }
  const double energy_flux =
      square(m_mode * omega) * 0.03125 *
      definite_integral(
        (square(abs(get<2, 2>(field_on_face) -
                    get<3, 3>(field_on_face))) +
        4. * square(abs(get<2, 3>(field_on_face)))) *
        get(face_jacobian) * integrand_multiplier,
          mesh.slice_away(0));
  const double surface_area = definite_integral(
      integrand_multiplier * get(face_jacobian), mesh.slice_away(0));
  return {energy_flux, surface_area};
}

}  // namespace GrSelfForce::Events::detail
