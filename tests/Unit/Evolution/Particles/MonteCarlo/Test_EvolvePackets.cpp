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
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"

using hydro::units::nuclear::proton_mass;

namespace {

void test_evolve_minkowski() {
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
tnsr::I<DataVector, 3, Frame::ElementLogical> spatial_coords_logical(
   const Mesh<3>& mesh) {
  const DataVector used_for_size(mesh.number_of_grid_points());
  auto x = make_with_value<tnsr::I<DataVector, 3, Frame::ElementLogical>>(
      used_for_size, 0.0);
  const Index<3>& extents = mesh.extents();
  Index<3> cur_extents{0, 0, 0};
  for (cur_extents[0] = 0; cur_extents[0] < extents[0]; cur_extents[0]++) {
    for (cur_extents[1] = 0; cur_extents[1] < extents[1]; cur_extents[1]++) {
      for (cur_extents[2] = 0; cur_extents[2] < extents[2]; cur_extents[2]++) {
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
  // inertial coordinates are set such that the element used for the
  // evolution is centered on (6,0,0), and the jacobian matrix
  // is the identity matrix.
tnsr::I<DataVector, 3, Frame::Inertial> spatial_coords_inertial(
    tnsr::I<DataVector, 3, Frame::ElementLogical> logical_coords) {
  auto x = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      logical_coords.get(0), 0.0);
  x.get(0) = logical_coords.get(0) + 6.0;
  x.get(1) = logical_coords.get(1);
  x.get(2) = logical_coords.get(2);
  return x;
}
  // Evolve a single Monte-Carlo packet along Kerr geodesic
  // with check that final position matches analytical expectations
void test_evolve_kerr(const size_t mesh_size_1d) {
  MAKE_GENERATOR(generator);

  // Parameters for KerrSchild solution
  const double mass = 1.01;
  const std::array<double, 3> spin{{0.0, 0.0, 0.0}};
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};
  const double t = 1.3;
  // Evaluate solution
  gr::Solutions::KerrSchild solution(mass, spin, center);

  // Set domain and coordintes
  const Mesh<3> mesh(mesh_size_1d, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered);
  const DataVector zero_dv(mesh.number_of_grid_points(), 0.0);
  const auto mesh_coordinates = spatial_coords_logical(mesh);
  const auto inertial_coordinates = spatial_coords_inertial(mesh_coordinates);
  const size_t num_ghost_zones = 1;
  const size_t extended_mesh_1d_size = mesh_size_1d + 2 * num_ghost_zones;
  const Mesh<3> extended_mesh(extended_mesh_1d_size,
                              Spectral::Basis::FiniteDifference,
                              Spectral::Quadrature::CellCentered);

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
  Scalar<DataVector> cell_light_crossing_time =
      make_with_value<Scalar<DataVector>>(zero_dv, 1.0);
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

  // Initialize packet on the x-axis
  const double dx = 2.0 / mesh_size_1d;
  Particles::MonteCarlo::Packet packet(0, 1.0, 5, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0,
                                       1.0, 0.0);
  // Self-consistency: update index of closest point and p^t
  std::array<size_t, 3> closest_point_index_3d{0, 0, 0};
  for (size_t d = 0; d < 3; d++) {
    gsl::at(closest_point_index_3d, d) =
        std::floor((packet.coordinates[d] - mesh_coordinates.get(d)[0]) /
                       (2.0 / mesh_size_1d) +
                   0.5);
  }
  const Index<3>& extents = mesh.extents();
  packet.index_of_closest_grid_point =
      closest_point_index_3d[0] +
      extents[0] *
          (closest_point_index_3d[1] + extents[1] * closest_point_index_3d[2]);
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

  Scalar<DataVector> coupling_tilde_tau =
      make_with_value<Scalar<DataVector>>(zero_dv, 0.0);
  Scalar<DataVector> coupling_rho_ye =
      make_with_value<Scalar<DataVector>>(zero_dv, 0.0);
  tnsr::i<DataVector, 3, Frame::Inertial> coupling_tilde_s =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(zero_dv, 0.0);

  std::vector<Particles::MonteCarlo::Packet> packets{packet};
  Particles::MonteCarlo::TemplatedLocalFunctions<2, 2> MonteCarloStruct;
  // Getting CFL constant
  const double cfl_constant = 0.25;
  const double final_time = 1.0;
  const double dt_step = cfl_constant * dx;
  double current_time = 0.0;
  packets[0].renormalize_momentum(inverse_spatial_metric, lapse);

  //time evolution with step size dt
  while (current_time < final_time) {
    double dt = std::min(dt_step, final_time - current_time);
    MonteCarloStruct.evolve_packets(
        &packets, &generator, &coupling_tilde_tau, &coupling_tilde_s,
        &coupling_rho_ye, current_time + dt, mesh, mesh_coordinates,
        num_ghost_zones, absorption_opacity, scattering_opacity,
        energy_at_bin_center, lorentz_factor, lower_spatial_four_velocity,
        lapse, shift, deriv_lapse, deriv_shift, deriv_inverse_spatial_metric,
        spatial_metric, inverse_spatial_metric, cell_light_crossing_time,
        mesh_velocity, inverse_jacobian, jacobian_inertial_to_fluid,
        inverse_jacobian_inertial_to_fluid);
    current_time += dt;
    packets[0].renormalize_momentum(inverse_spatial_metric, lapse);
  }
  const double final_r = sqrt(pow(packets[0].coordinates.get(0) + 6.0, 2) +
   pow(packets[0].coordinates.get(1), 2)+pow(packets[0].coordinates.get(2),2));
  //Check r against geodesic evolution value obtained from python code
  CHECK(std::abs(final_r - 6.298490) < 1e-2);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Particles.MonteCarloEvolution",
                  "[Unit][Evolution]") {
  test_evolve_minkowski();
  test_evolve_kerr(7);
}
