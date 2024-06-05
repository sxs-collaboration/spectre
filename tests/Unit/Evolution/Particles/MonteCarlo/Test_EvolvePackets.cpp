// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Particles/MonteCarlo/EvolvePackets.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Evolution/Particles/MonteCarlo/TemplatedLocalFunctions.hpp"
#include "Framework/TestHelpers.hpp"
#include "Framework/TestingFramework.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"

using hydro::units::nuclear::proton_mass;

namespace {

void test_evolve_minkowski() {
  const Mesh<3> mesh(2, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered);

  MAKE_GENERATOR(generator);

  const size_t dv_size = 8;
  DataVector zero_dv(dv_size, 0.0);
  // Minkowslo metric
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
  tnsr::I<DataVector, 3, Frame::Inertial> inertial_coordinates =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  inertial_coordinates.get(0) =
      DataVector{-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5};
  inertial_coordinates.get(1) =
      DataVector{-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5};
  inertial_coordinates.get(2) =
      DataVector{-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5};

  Particles::MonteCarlo::Packet packet(1, 1.0, 0, 0.0, -0.75, -1.0, -1.0, 1.0,
                                       1.0, 0.0, 0.0);

  packet.renormalize_momentum(inv_spatial_metric, lapse);
  CHECK(packet.momentum_upper_t == 1.0);

  const std::array<double, 2> energy_at_bin_center = {2.0, 5.0};
  std::array<std::array<DataVector, 2>, 2> absorption_opacity = {
      std::array<DataVector, 2>{{zero_dv, zero_dv}},
      std::array<DataVector, 2>{{zero_dv, zero_dv}}};
  std::array<std::array<DataVector, 2>, 2> scattering_opacity = {
      std::array<DataVector, 2>{{zero_dv, zero_dv}},
      std::array<DataVector, 2>{{zero_dv, zero_dv}}};

  // Set non-zero value that should never lead
  // to interaction, to get non-zero interaction terms
  // (the choices made for the minimum value of the
  //  random number setting the time to next interaction
  //  guarantees that for such low opacities, interactions
  //  will not happen).
  gsl::at(gsl::at(absorption_opacity, 1), 0) = 1.e-60;
  gsl::at(gsl::at(scattering_opacity, 1), 0) = 1.e-59;

  Scalar<DataVector> coupling_tilde_tau =
      make_with_value<Scalar<DataVector>>(zero_dv, 0.0);
  Scalar<DataVector> coupling_rho_ye =
      make_with_value<Scalar<DataVector>>(zero_dv, 0.0);
  tnsr::i<DataVector, 3, Frame::Inertial> coupling_tilde_s =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(zero_dv, 0.0);

  std::vector<Particles::MonteCarlo::Packet> packets{packet};
  Particles::MonteCarlo::TemplatedLocalFunctions<2, 2> MonteCarloStruct;
  MonteCarloStruct.evolve_packets(
      &packets, &generator, &coupling_tilde_tau, &coupling_tilde_s,
      &coupling_rho_ye, 1.5, mesh, mesh_coordinates, inertial_coordinates,
      absorption_opacity, scattering_opacity, energy_at_bin_center,
      lorentz_factor, lower_spatial_four_velocity, lapse, shift,
      d_lapse, d_shift, d_inv_spatial_metric,
      spatial_metric, inv_spatial_metric, mesh_velocity,
      inverse_jacobian, jacobian_inertial_to_fluid,
      inverse_jacobian_inertial_to_fluid);
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
  CHECK(fabs(get(coupling_tilde_tau)[0] - 1.0e-60) < 1.5e-75);
  CHECK(fabs(coupling_tilde_s.get(0)[0] - 1.1e-59) < 1.65e-74);
  CHECK(coupling_tilde_s.get(1)[0] == 0.0);
  CHECK(coupling_tilde_s.get(2)[0] == 0.0);
  CHECK(fabs(get(coupling_rho_ye)[0] + proton_mass * 1.0e-60) < 1.e-72);
  CHECK(fabs(get(coupling_tilde_tau)[1] - 5.0e-61) < 1.5e-75);
  CHECK(fabs(coupling_tilde_s.get(0)[1] - 5.5e-60) < 1.65e-74);
  CHECK(coupling_tilde_s.get(1)[1] == 0.0);
  CHECK(coupling_tilde_s.get(2)[1] == 0.0);
  CHECK(fabs(get(coupling_rho_ye)[1] + proton_mass * 5.0e-61) < 1.e-72);
}

tnsr::I<DataVector, 3, Frame::ElementLogical> spatial_coords_logical(
    const Mesh<3>& mesh) {
  const DataVector used_for_size(mesh.number_of_grid_points());
  auto x = make_with_value<tnsr::I<DataVector, 3, Frame::ElementLogical>>(
      used_for_size, 0.0);
  const Index<3>& extents = mesh.extents();
  Index<3> cur_extents{0,0,0};
  for (cur_extents[0]=0 ; cur_extents[0] < extents[0]; cur_extents[0]++) {
    for (cur_extents[1]=0; cur_extents[1] < extents[1]; cur_extents[1]++) {
      for (cur_extents[2]=0; cur_extents[2] < extents[2]; cur_extents[2]++) {
        const size_t storage_index = mesh.storage_index(cur_extents);
        x.get(0)[storage_index] =
            -1.0 + 2.0 * static_cast<double>(cur_extents[0] + 0.5) /
                       static_cast<double>(extents[0]);
        x.get(1)[storage_index] =
            -1.0 + 2.0 * static_cast<double>(cur_extents[1] + 0.5) /
                       static_cast<double>(extents[1]);
        x.get(2)[storage_index] =
            -1.0 + 2.0 * static_cast<double>(cur_extents[2] + 0.5) /
                       static_cast<double>(extents[2]);
      }
    }
  }
  return x;
}
tnsr::I<DataVector, 3, Frame::Inertial> spatial_coords_inertial(
    tnsr::I<DataVector, 3, Frame::ElementLogical> logical_coords) {
  auto x = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      logical_coords.get(0), 0.0);
  x.get(0) = logical_coords.get(0) + 6.0;
  x.get(1) = logical_coords.get(1);
  x.get(2) = logical_coords.get(2);
  return x;
}
// Function to calculate p_phi
double calculate_p_phi(const Particles::MonteCarlo::Packet& packet) {
  return -packet.momentum.get(0) * packet.coordinates.get(1) +
         packet.momentum.get(1) * (packet.coordinates.get(0) + 6.0);
}

void test_evolve_kerr() {
  MAKE_GENERATOR(generator);

  // Parameters for KerrSchild solution
  const double mass = 1.01;
  const std::array<double, 3> spin{{0.0, 0.0, 0.0}};
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};
  const double t = 1.3;
  // Evaluate solution
  gr::Solutions::KerrSchild solution(mass, spin, center);

  // Set domain and coordintes
  const size_t mesh_size_1d = 15;
  const Mesh<3> mesh(mesh_size_1d, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered);
  const DataVector zero_dv(mesh.number_of_grid_points(),0.0);
  const auto mesh_coordinates = spatial_coords_logical(mesh);
  const auto inertial_coordinates = spatial_coords_inertial(mesh_coordinates);
  // Mesh velocity set to std::null for now
  const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> mesh_velocity =
      std::nullopt;

  // Compute metric quantities
  const auto vars = solution.variables(
      inertial_coordinates, t,
      typename gr::Solutions::KerrSchild::tags<DataVector, Frame::Inertial>{});
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(vars);
  const auto& deriv_lapse = get<typename gr::Solutions::KerrSchild::DerivLapse<
      DataVector, Frame::Inertial>>(vars);
  const auto& shift =
      get<gr::Tags::Shift<DataVector, 3, Frame::Inertial>>(vars);
  const auto& deriv_shift = get<typename gr::Solutions::KerrSchild::DerivShift<
      DataVector, Frame::Inertial>>(vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>>(vars);
  const auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>>(vars);
  const auto& deriv_spatial_metric =
      get<typename gr::Solutions::KerrSchild::DerivSpatialMetric<
          DataVector, Frame::Inertial>>(vars);

  tnsr::iJJ<DataVector, 3, Frame::Inertial> deriv_inverse_spatial_metric =
      make_with_value<tnsr::iJJ<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = i; j < 3; j++) {
      for (size_t k = 0; k < 3; k++) {
        for (size_t l = 0; l < 3; l++) {
          for (size_t m = 0; m < 3; m++) {
            deriv_inverse_spatial_metric.get(k, i, j) -=
                inverse_spatial_metric.get(i, l) *
                inverse_spatial_metric.get(j, m) *
                deriv_spatial_metric.get(k, l, m);
          }
        }
      }
    }
  }

  // Fluid quantities
  Scalar<DataVector> lorentz_factor =
      make_with_value<Scalar<DataVector>>(lapse, 1.0);
  tnsr::i<DataVector, 3, Frame::Inertial> lower_spatial_four_velocity =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(lapse, 0.0);

  // Initialize packet on the x-axis, moving in the y-direction
  const double dx = 2.0/mesh_size_1d;
  Particles::MonteCarlo::Packet packet(0, 1.0, 5, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0,
                                       1.0, 0.0);
  // Self-consistency: update index of closest point and p^t
  std::array<size_t, 3> closest_point_index_3d{0, 0, 0};
  for (size_t d = 0; d < 3; d++) {
    gsl::at(closest_point_index_3d, d) =
      std::floor((packet.coordinates[d] - mesh_coordinates.get(d)[0]) /
        (2.0/mesh_size_1d) + 0.5);
  }
  const Index<3>& extents = mesh.extents();
  packet.index_of_closest_grid_point =
      closest_point_index_3d[0] +
        extents[0] * (closest_point_index_3d[1] +
                    extents[1] * closest_point_index_3d[2]);
  packet.renormalize_momentum(inverse_spatial_metric, lapse);

  // Jacobians set to identity for now
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

  // Interaction rates
  const std::array<double, 2> energy_at_bin_center = {2.0, 5.0};
  std::array<std::array<DataVector, 2>, 2> absorption_opacity = {
      std::array<DataVector, 2>{{zero_dv, zero_dv}},
      std::array<DataVector, 2>{{zero_dv, zero_dv}}};
  std::array<std::array<DataVector, 2>, 2> scattering_opacity = {
      std::array<DataVector, 2>{{zero_dv, zero_dv}},
      std::array<DataVector, 2>{{zero_dv, zero_dv}}};

  Scalar<DataVector> coupling_tilde_tau
    = make_with_value< Scalar<DataVector> >(zero_dv, 0.0);
  Scalar<DataVector> coupling_rho_ye
    = make_with_value< Scalar<DataVector> >(zero_dv, 0.0);
  tnsr::i<DataVector, 3, Frame::Inertial> coupling_tilde_s
    = make_with_value< tnsr::i<DataVector, 3, Frame::Inertial> >(zero_dv, 0.0);

  std::vector<Particles::MonteCarlo::Packet> packets{packet};
  Particles::MonteCarlo::TemplatedLocalFunctions<2, 2> MonteCarloStruct;
  // Getting N and CFL constant
  const double cfl_constant = 0.25;
  const double final_time = 1.0;
  const double dt_step = cfl_constant *dx;
  double current_time = 0.0;

  // Store initial value
  double initial_p_phi = calculate_p_phi(packet);
  Parallel::printf("Initial p_phi: %.5f\n", initial_p_phi);

  Parallel::printf("%.5f %.5f %.5f %.5e %.2d\n", packets[0].time,
                   packets[0].coordinates.get(0), packets[0].coordinates.get(1),
                   packets[0].coordinates.get(2),
                   packets[0].index_of_closest_grid_point);
  while (current_time < final_time) {
    double dt = std::min(dt_step, final_time - current_time);
    MonteCarloStruct.evolve_packets(
        &packets, &generator, &coupling_tilde_tau, &coupling_tilde_s,
        &coupling_rho_ye, current_time + dt, mesh, mesh_coordinates,
        absorption_opacity, scattering_opacity, energy_at_bin_center,
        lorentz_factor, lower_spatial_four_velocity, lapse, shift, deriv_lapse,
        deriv_shift, deriv_inverse_spatial_metric, spatial_metric,
        inverse_spatial_metric, mesh_velocity, inverse_jacobian,
        jacobian_inertial_to_fluid, inverse_jacobian_inertial_to_fluid);
    current_time += dt;
    Parallel::printf(
        "%.5f %.5f %.5f %.5e %.2d\n", packets[0].time,
        packets[0].coordinates.get(0), packets[0].coordinates.get(1),
        packets[0].coordinates.get(2), packets[0].index_of_closest_grid_point);
    double p_phi = calculate_p_phi(packets[0]);
    const size_t c_idx = packets[0].index_of_closest_grid_point;
    Parallel::printf(
      "Closest Pt: %.5f %.5f %.5f\n", mesh_coordinates.get(0)[c_idx],
      mesh_coordinates.get(1)[c_idx], mesh_coordinates.get(2)[c_idx]);
    Parallel::printf("p_phi: %.5f\n", p_phi);
  }
  Parallel::printf("%.5f %.5f %.5f %.5e \n", packets[0].time,
                   packets[0].coordinates.get(0), packets[0].coordinates.get(1),
                   packets[0].coordinates.get(2));

  double final_p_phi = calculate_p_phi(packets[0]);
  Parallel::printf("Final p_phi: %.5f\n", final_p_phi);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Particles.MonteCarloEvolution",
                  "[Unit][Evolution]") {
  test_evolve_minkowski();
  test_evolve_kerr();
}
