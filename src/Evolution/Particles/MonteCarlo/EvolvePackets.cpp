// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Particles/MonteCarlo/EvolvePackets.hpp"

#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Evolution/Particles/MonteCarlo/Scattering.hpp"
#include "Parallel/Printf/Printf.hpp"

namespace Particles::MonteCarlo {

namespace detail {

// Time derivative of p_i in a packet, according to geodesic equation
void time_derivative_momentum_geodesic(
    const gsl::not_null<std::array<double, 3>*> dt_momentum,
    const Packet& packet, const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
    const tnsr::iJJ<DataVector, 3, Frame::Inertial>& d_inv_spatial_metric) {
  const size_t& closest_point_index = packet.index_of_closest_grid_point;
  for (size_t i = 0; i < 3; i++) {
    gsl::at(*dt_momentum, i) = (-1.0) * d_lapse.get(i)[closest_point_index] *
                               get(lapse)[closest_point_index] *
                               packet.momentum_upper_t;
    for (size_t j = 0; j < 3; j++) {
      gsl::at(*dt_momentum, i) +=
          d_shift.get(i, j)[closest_point_index] * packet.momentum[j];
      for (size_t k = 0; k < 3; k++) {
        gsl::at(*dt_momentum, i) -=
            0.5 * d_inv_spatial_metric.get(i, j, k)[closest_point_index] *
            packet.momentum[j] * packet.momentum[k] / packet.momentum_upper_t;
      }
    }
  }
}

}  // namespace detail

void evolve_single_packet_on_geodesic(
    const gsl::not_null<Packet*> packet, const double time_step,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
    const tnsr::iJJ<DataVector, 3, Frame::Inertial>& d_inv_spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>& mesh_velocity,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>&
        inverse_jacobian_logical_to_inertial) {
  const size_t& closest_point_index = packet->index_of_closest_grid_point;
  // Temporary variables
  std::array<double, 3> dpdt{0, 0, 0};
  std::array<double, 3> dxdt_inertial{0, 0, 0};
  std::array<double, 3> dxdt_logical{0, 0, 0};
  std::array<double, 3> p0{0, 0, 0};

  // Calculate p^t from normalization of 4-momentum
  packet->renormalize_momentum(inv_spatial_metric, lapse);

  // Calculate time derivative of 3-momentum at beginning of time step
  detail::time_derivative_momentum_geodesic(&dpdt, *packet, lapse, d_lapse,
                                            d_shift, d_inv_spatial_metric);
  // Parallel::printf("Initial coord: %.5f %.5f %.5f\n", packet->coordinates[0],
  //                  packet->coordinates[1], packet->coordinates[2]);
  // Parallel::printf("Initial momentum: %.5f %.5f %.5f\n", packet->momentum[0],
  //                  packet->momentum[1], packet->momentum[2]);
  // Parallel::printf("dpdt: %.5f %.5f %.5f\n", gsl::at(dpdt, 0), gsl::at(dpdt,
  // 1),
  //                  gsl::at(dpdt, 2));

  // Take half-step (momentum only, as time derivative is independent of
  // position)
  for (size_t i = 0; i < 3; i++) {
    // Store momentum at beginning of step
    gsl::at(p0, i) = packet->momentum[i];
    packet->momentum[i] += gsl::at(dpdt, i) * time_step * 0.5;
  }
  // Calculate p^t from normalization of 4-momentum
  packet->renormalize_momentum(inv_spatial_metric, lapse);

  // Calculate time derivative of 3-momentum and position at half-step
  detail::time_derivative_momentum_geodesic(&dpdt, *packet, lapse, d_lapse,
                                            d_shift, d_inv_spatial_metric);
  // Parallel::printf("Half step momentum: %.5f %.5f %.5f\n",
  // packet->momentum[0],
  //                 packet->momentum[1], packet->momentum[2]);
  // Parallel::printf("dpdt: %.5f %.5f %.5f\n", gsl::at(dpdt, 0), gsl::at(dpdt,
  // 1),
  //                 gsl::at(dpdt, 2));
  for (size_t i = 0; i < 3; i++) {
    gsl::at(dxdt_inertial, i) = (-1.0) * shift.get(i)[closest_point_index];
    if (mesh_velocity.has_value()) {
      gsl::at(dxdt_inertial, i) -=
          mesh_velocity.value().get(i)[closest_point_index];
    }
    for (size_t j = 0; j < 3; j++) {
      gsl::at(dxdt_inertial, i) +=
          inv_spatial_metric.get(i, j)[closest_point_index] *
          packet->momentum[j] / packet->momentum_upper_t;
    }
  }
  // Parallel::printf("dxdt_inert: %.5f %.5f %.5f\n", gsl::at(dxdt_inertial, 0),
  //                 gsl::at(dxdt_inertial, 1), gsl::at(dxdt_inertial, 2));

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      gsl::at(dxdt_logical, i) +=
          gsl::at(dxdt_inertial, j) *
          inverse_jacobian_logical_to_inertial.get(i, j)[closest_point_index];
    }
  }
  // Parallel::printf("dxdt_inert: %.5f %.5f %.5f\n", gsl::at(dxdt_logical, 0),
  //                 gsl::at(dxdt_logical, 1), gsl::at(dxdt_logical, 2));

  // Take full time step
  for (size_t i = 0; i < 3; i++) {
    packet->momentum[i] = gsl::at(p0, i) + gsl::at(dpdt, i) * time_step;
    packet->coordinates[i] += gsl::at(dxdt_logical, i) * time_step;
  }
  packet->time += time_step;
  // Parallel::printf("Final coord: %.5f %.5f %.5f\n", packet->coordinates[0],
  //                 packet->coordinates[1], packet->coordinates[2]);
  // Parallel::printf("Final momentum: %.5f %.5f %.5f\n", packet->momentum[0],
  //                  packet->momentum[1], packet->momentum[2]);
}

}  // namespace Particles::MonteCarlo
