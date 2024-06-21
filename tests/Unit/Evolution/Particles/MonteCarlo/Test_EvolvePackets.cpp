// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Particles/MonteCarlo/EvolvePackets.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Evolution/Particles/MonteCarlo/TemplatedLocalFunctions.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"

using hydro::units::nuclear::proton_mass;

SPECTRE_TEST_CASE("Unit.Evolution.Particles.MonteCarloEvolution",
                  "[Unit][Evolution]") {
  const size_t mesh_1d_size = 2;
  const Mesh<3> mesh(mesh_1d_size, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered);
  const size_t num_ghost_zones = 1;
  const size_t extended_mesh_1d_size = mesh_1d_size + 2 * num_ghost_zones;
  const Mesh<3> extended_mesh(extended_mesh_1d_size,
                              Spectral::Basis::FiniteDifference,
                              Spectral::Quadrature::CellCentered);

  MAKE_GENERATOR(generator);

  const size_t dv_size = mesh.number_of_grid_points();
  const size_t extended_dv_size = extended_mesh.number_of_grid_points();
  DataVector zero_dv(dv_size, 0.0);
  DataVector extended_zero_dv(extended_dv_size, 0.0);
  // Minkowski metric
  Scalar<DataVector> lapse{DataVector(dv_size, 1.0)};
  Scalar<DataVector> lorentz_factor{
      DataVector{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};

  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  inv_spatial_metric.get(0, 0) = 1.0;
  inv_spatial_metric.get(1, 1) = 1.0;
  inv_spatial_metric.get(2, 2) = 1.0;
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  spatial_metric.get(0, 0) = 1.0;
  spatial_metric.get(1, 1) = 1.0;
  spatial_metric.get(2, 2) = 1.0;

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
  mesh_coordinates.get(0) =
      DataVector{-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5};
  mesh_coordinates.get(1) =
      DataVector{-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5};
  mesh_coordinates.get(2) =
      DataVector{-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5};
  Scalar<DataVector> cell_light_crossing_time =
      make_with_value<Scalar<DataVector>>(zero_dv, 1.0);

  Particles::MonteCarlo::Packet packet(1, 1.0, 0, 0.0, -0.75, -1.0, -1.0, 1.0,
                                       1.0, 0.0, 0.0);

  packet.renormalize_momentum(inv_spatial_metric, lapse);
  CHECK(packet.momentum_upper_t == 1.0);

  const std::array<double, 2> energy_at_bin_center = {2.0, 5.0};
  std::array<std::array<DataVector, 2>, 2> absorption_opacity = {
      std::array<DataVector, 2>{{extended_zero_dv, extended_zero_dv}},
      std::array<DataVector, 2>{{extended_zero_dv, extended_zero_dv}}};
  std::array<std::array<DataVector, 2>, 2> scattering_opacity = {
      std::array<DataVector, 2>{{extended_zero_dv, extended_zero_dv}},
      std::array<DataVector, 2>{{extended_zero_dv, extended_zero_dv}}};

  // Set non-zero value that should never lead
  // to interaction, to get non-zero interaction terms
  // (the choices made for the minimum value of the
  //  random number setting the time to next interaction
  //  guarantees that for such low opacities, interactions
  //  will not happen).
  gsl::at(gsl::at(absorption_opacity, 1), 0) = 1.e-60;
  gsl::at(gsl::at(scattering_opacity, 1), 0) = 1.e-59;

  Scalar<DataVector> coupling_tilde_tau =
      make_with_value<Scalar<DataVector>>(extended_zero_dv, 0.0);
  Scalar<DataVector> coupling_rho_ye =
      make_with_value<Scalar<DataVector>>(extended_zero_dv, 0.0);
  tnsr::i<DataVector, 3, Frame::Inertial> coupling_tilde_s =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(extended_zero_dv,
                                                               0.0);
  std::vector<Particles::MonteCarlo::Packet> packets{packet};
  Particles::MonteCarlo::TemplatedLocalFunctions<2, 2> MonteCarloStruct;
  MonteCarloStruct.evolve_packets(
      &packets, &generator, &coupling_tilde_tau, &coupling_tilde_s,
      &coupling_rho_ye, 1.5, mesh, mesh_coordinates, num_ghost_zones,
      absorption_opacity, scattering_opacity, energy_at_bin_center,
      lorentz_factor, lower_spatial_four_velocity, lapse, shift, d_lapse,
      d_shift, d_inv_spatial_metric, spatial_metric, inv_spatial_metric,
      cell_light_crossing_time, mesh_velocity, inverse_jacobian,
      jacobian_inertial_to_fluid, inverse_jacobian_inertial_to_fluid);
  CHECK(packets[0].species == 1);
  CHECK(packets[0].coordinates.get(0) == 0.75);
  CHECK(packets[0].coordinates.get(1) == -1.0);
  CHECK(packets[0].coordinates.get(2) == -1.0);
  CHECK(packets[0].momentum.get(0) == 1.0);
  CHECK(packets[0].momentum.get(1) == 0.0);
  CHECK(packets[0].momentum.get(2) == 0.0);
  CHECK(packets[0].time == 1.5);
  CHECK(packets[0].index_of_closest_grid_point == 1);
  // Check coupling terms against analytical expectations
  // Note that the packet spends dt = 1.0 in cell 0 and
  // dt=0.5 in cell 1 (due to the use of partial time steps)
  // and that we need to convert these 'live points' index
  // to the mesh with ghost zones.
  const size_t ext_idx_0 =
      num_ghost_zones + num_ghost_zones * extended_mesh_1d_size +
      num_ghost_zones * extended_mesh_1d_size * extended_mesh_1d_size;
  CHECK(fabs(get(coupling_tilde_tau)[ext_idx_0] - 1.0e-60) < 1.5e-75);
  CHECK(fabs(coupling_tilde_s.get(0)[ext_idx_0] - 1.1e-59) < 1.65e-74);
  CHECK(coupling_tilde_s.get(1)[ext_idx_0] == 0.0);
  CHECK(coupling_tilde_s.get(2)[ext_idx_0] == 0.0);
  CHECK(fabs(get(coupling_rho_ye)[ext_idx_0] + proton_mass * 1.0e-60) < 1.e-72);
  CHECK(fabs(get(coupling_tilde_tau)[ext_idx_0 + 1] - 5.0e-61) < 1.5e-75);
  CHECK(fabs(coupling_tilde_s.get(0)[ext_idx_0 + 1] - 5.5e-60) < 1.65e-74);
  CHECK(coupling_tilde_s.get(1)[ext_idx_0 + 1] == 0.0);
  CHECK(coupling_tilde_s.get(2)[ext_idx_0 + 1] == 0.0);
  CHECK(fabs(get(coupling_rho_ye)[ext_idx_0 + 1] + proton_mass * 5.0e-61)
    < 1.e-72);
}
