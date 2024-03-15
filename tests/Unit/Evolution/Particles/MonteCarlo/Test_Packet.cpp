// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Particles/MonteCarlo/EvolvePackets.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace {

void check_packet() {
  const double epsilon_approx = 1.e-15;

  const size_t dv_size = 5;
  Scalar<DataVector> lapse(dv_size, 1.0);
  Scalar<DataVector> lorentz_factor(dv_size, 1.0);
  tnsr::i<DataVector, 3, Frame::Inertial> lower_spatial_four_velocity =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(lapse, 0.0);

  const size_t pos = 3;
  Particles::MonteCarlo::Packet packet(1, 2.0, pos, 0.0, -1.0, -0.5, -0.2, 0.9,
                                       -0.3, 0.2, 0.1);
  // The values below are wildly inconsistent, but we only want to check the
  // math
  get(lapse)[pos] = 0.8;
  get(lorentz_factor)[pos] = 1.3;
  lower_spatial_four_velocity.get(0)[pos] = 0.3;
  lower_spatial_four_velocity.get(1)[pos] = 0.4;
  lower_spatial_four_velocity.get(2)[pos] = 0.5;
  inv_spatial_metric.get(0, 0) = 1.1;
  inv_spatial_metric.get(1, 1) = 1.2;
  inv_spatial_metric.get(2, 2) = 1.3;
  inv_spatial_metric.get(0, 1) = 0.1;
  inv_spatial_metric.get(0, 2) = 0.2;
  inv_spatial_metric.get(1, 2) = 0.3;

  // p^t = \sqrt{\gamma^{ij} p_i p_j}/\alpha
  const double expected_momentum_upper_t =
      sqrt(1.1 * 0.3 * 0.3 + 1.2 * 0.2 * 0.2 + 1.3 * 0.1 * 0.1 +
           2.0 * 0.1 * (-0.3) * 0.2 + 2.0 * 0.2 * (-0.3) * 0.1 +
           2.0 * 0.3 * 0.2 * 0.1) /
      0.8;
  packet.renormalize_momentum(inv_spatial_metric, lapse);
  CHECK(fabs(packet.momentum_upper_t - expected_momentum_upper_t) <
        epsilon_approx);

  // E = W \alpha p^t - \gamma^{ij} u_i p_j
  const double expected_energy =
      0.8 * 1.3 * expected_momentum_upper_t - 1.1 * (-0.3) * 0.3 -
      1.2 * 0.2 * 0.4 - 1.3 * 0.1 * 0.5 - 0.1 * (-0.3) * 0.4 -
      0.1 * (0.2) * (0.3) - 0.2 * (-0.3) * 0.5 - 0.2 * 0.1 * 0.3 -
      0.3 * 0.2 * 0.5 - 0.3 * 0.1 * 0.4;
  const double computed_energy =
      Particles::MonteCarlo::compute_fluid_frame_energy(
          packet, lorentz_factor, lower_spatial_four_velocity, lapse,
          inv_spatial_metric);
  CHECK(fabs(expected_energy - computed_energy) < epsilon_approx);

  CHECK(packet.species == 1);
  CHECK(packet.coordinates.get(0) == -1.0);
  CHECK(packet.coordinates.get(1) == -0.5);
  CHECK(packet.coordinates.get(2) == -0.2);
  CHECK(packet.momentum.get(0) == -0.3);
  CHECK(packet.momentum.get(1) == 0.2);
  CHECK(packet.momentum.get(2) == 0.1);
  CHECK(packet.time == 0.0);
  CHECK(packet.index_of_closest_grid_point == pos);
  CHECK(packet.number_of_neutrinos == 2.0);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Particles.MonteCarlo", "[Unit][Evolution]") {
  check_packet();

  const Mesh<3> mesh(2, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered);

  MAKE_GENERATOR(generator);

  // Minkowski metric
  Scalar<DataVector> lapse{DataVector{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};
  Scalar<DataVector> lorentz_factor{
      DataVector{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};

  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  inv_spatial_metric.get(0, 0) = 1.0;
  inv_spatial_metric.get(1, 1) = 1.0;
  inv_spatial_metric.get(2, 2) = 1.0;

  tnsr::I<DataVector, 3, Frame::Inertial> shift =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  tnsr::i<DataVector, 3, Frame::Inertial> lower_spatial_four_velocity =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  tnsr::i<DataVector, 3, Frame::Inertial> d_lapse =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  tnsr::iJ<DataVector, 3, Frame::Inertial> d_shift =
      make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  tnsr::iJJ<DataVector, 3, Frame::Inertial> d_inv_spatial_metric =
      make_with_value<tnsr::iJJ<DataVector, 3, Frame::Inertial>>(lapse, 0.0);

  // Mesh velocity set to std::null for now
  const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> mesh_velocity =
      std::nullopt;

  // Jacobian set to identity for now
  InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>
      inverse_jacobian_inertial_to_fluid = make_with_value<
          InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>>(lapse,
                                                                         0.0);
  inverse_jacobian_inertial_to_fluid.get(0, 0) = 1.0;
  inverse_jacobian_inertial_to_fluid.get(1, 1) = 1.0;
  inverse_jacobian_inertial_to_fluid.get(2, 2) = 1.0;
  inverse_jacobian_inertial_to_fluid.get(3, 3) = 1.0;
  Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>
      jacobian_inertial_to_fluid = make_with_value<
          Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>>(lapse, 0.0);
  jacobian_inertial_to_fluid.get(0, 0) = 1.0;
  jacobian_inertial_to_fluid.get(1, 1) = 1.0;
  jacobian_inertial_to_fluid.get(2, 2) = 1.0;
  jacobian_inertial_to_fluid.get(3, 3) = 1.0;

  // Logical to inertial inverse jacobian, also identity for now
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      inverse_jacobian =
          make_with_value<InverseJacobian<DataVector, 3, Frame::ElementLogical,
                                          Frame::Inertial>>(lapse, 0.0);
  inverse_jacobian.get(0, 0) = 1.0;
  inverse_jacobian.get(1, 1) = 1.0;
  inverse_jacobian.get(2, 2) = 1.0;

  // Coordinates
  tnsr::I<DataVector, 3, Frame::ElementLogical> mesh_coordinates =
      make_with_value<tnsr::I<DataVector, 3, Frame::ElementLogical>>(lapse,
                                                                     0.0);
  mesh_coordinates.get(0) = DataVector{-1., 1., -1., 1., -1., 1., -1., 1.};
  mesh_coordinates.get(1) = DataVector{-1., -1., 1., 1., -1., -1., 1., 1.};
  mesh_coordinates.get(2) = DataVector{-1., -1., -1., -1., 1., 1., 1., 1.};

  Particles::MonteCarlo::Packet packet(2, 1.0, 0, 0.0, -1.0, -1.0, -1.0, 1.0,
                                       1.0, 0.0, 0.0);

  packet.renormalize_momentum(inv_spatial_metric, lapse);
  CHECK(packet.momentum_upper_t == 1.0);

  std::vector<Particles::MonteCarlo::Packet> packets{packet};
  Particles::MonteCarlo::evolve_packets(
    &packets, &generator, 1.5, mesh, mesh_coordinates, lorentz_factor,
    lower_spatial_four_velocity, lapse, shift, d_lapse, d_shift,
    d_inv_spatial_metric, inv_spatial_metric, mesh_velocity,
    inverse_jacobian, jacobian_inertial_to_fluid,
    inverse_jacobian_inertial_to_fluid);
  CHECK(packets[0].species == 2);
  CHECK(packets[0].coordinates.get(0) == 0.5);
  CHECK(packets[0].coordinates.get(1) == -1.0);
  CHECK(packets[0].coordinates.get(2) == -1.0);
  CHECK(packets[0].momentum.get(0) == 1.0);
  CHECK(packets[0].momentum.get(1) == 0.0);
  CHECK(packets[0].momentum.get(2) == 0.0);
  CHECK(packets[0].time == 1.5);
  CHECK(packets[0].index_of_closest_grid_point == 1);
}
