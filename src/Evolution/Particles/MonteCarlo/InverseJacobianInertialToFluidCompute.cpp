// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Particles/MonteCarlo/InverseJacobianInertialToFluidCompute.hpp"

#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace Particles::MonteCarlo {

void InverseJacobianInertialToFluidCompute::function(
    gsl::not_null<return_type*> inv_jacobian,
    const tnsr::I<DataVector, 3>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::ii<DataVector, 3>& spatial_metric) {
  // The components of the inverse jacobian are just the vector components
  // of the orthonormal tetrad comoving with the fluid in the inertial frame.

  // First, inv_jacobian(a,0) is just u^a
  inv_jacobian->get(0, 0) = get(lorentz_factor) / get(lapse);
  for (size_t d = 0; d < 3; d++) {
    inv_jacobian->get(d + 1, 0) =
        get(lorentz_factor) *
        (spatial_velocity.get(d) - shift.get(d) / get(lapse));
  }

  // Then, the other members of the tetrad are constructed using Gram-Schmidt

  // Temporary memory allocation
  auto temp_dot_product = make_with_value<DataVector>(lapse, 0.0);
  tnsr::aa<DataVector, 3, Frame::Inertial> spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);

  // d = 1,2,3 for tetrad components built from x,y,z
  for (size_t d = 1; d < 4; d++) {
    // Base vector for Gram-Shmidt
    for (size_t i = 0; i < 4; i++) {
      set_number_of_grid_points(make_not_null(&inv_jacobian->get(i, d)), lapse);
      inv_jacobian->get(i, d) = (i==d ? 1.0 : 0.0);
    }
    // Projection orthogonal to u^\mu
    temp_dot_product = 0.0;
    for (size_t i = 0; i < 3; i++) {
      temp_dot_product +=
          spatial_metric.get(d - 1, i) * spatial_velocity.get(i);
    }
    temp_dot_product *= get(lorentz_factor);
    // Note: + sign when projecting here, because the 0th tetrad vector has norm
    // -1.
    for (size_t i = 0; i < 4; i++) {
      inv_jacobian->get(i, d) += temp_dot_product * inv_jacobian->get(i, 0);
    }

    // Loop over other existing tetrad vectors to get orthogonal projection
    for (size_t a = 0; a < d; a++) {
      temp_dot_product = 0.0;
      for (size_t b = 0; b < 4; b++) {
        for (size_t c = 0; c < 4; c++) {
          temp_dot_product += spacetime_metric.get(b, c) *
                              inv_jacobian->get(b, d) * inv_jacobian->get(c, a);
        }
      }
      for (size_t i = 0; i < 4; i++) {
        inv_jacobian->get(i, d) -= temp_dot_product * inv_jacobian->get(i, a);
      }
    }

    // Normalize tetrad vector
    temp_dot_product = 0.0;
    for (size_t a = 0; a < 4; a++) {
      temp_dot_product += spacetime_metric.get(a, a) * inv_jacobian->get(a, d) *
                          inv_jacobian->get(a, d);
      for (size_t b = a + 1; b < 4; b++) {
        temp_dot_product += 2.0 * spacetime_metric.get(a, b) *
                            inv_jacobian->get(a, d) * inv_jacobian->get(b, d);
      }
    }
    temp_dot_product = sqrt(temp_dot_product);
    for (size_t a = 0; a < 4; a++) {
      inv_jacobian->get(a, d) = inv_jacobian->get(a, d) / temp_dot_product;
    }
  }
}

}  // namespace Particles::MonteCarlo
