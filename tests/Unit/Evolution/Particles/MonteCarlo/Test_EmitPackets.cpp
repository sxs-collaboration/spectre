// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Evolution/Particles/MonteCarlo/TemplatedLocalFunctions.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"

using hydro::units::nuclear::proton_mass;

SPECTRE_TEST_CASE("Unit.Evolution.Particles.MonteCarloEmission",
                  "[Unit][Evolution]") {
  using Particles::MonteCarlo::Packet;

  const double epsilon_approx = 1.e-13;
  // Vector of MC packets
  std::vector<Packet> all_packets = {};

  const double time = 0.0;
  const double time_step = 0.1;

  MAKE_GENERATOR(generator);
  const Mesh<3> mesh(3, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered);

  // Zero vector for tensor creations
  DataVector zero_dv(27, 0.0);

  std::array<std::array<DataVector, 2>, 2> emission_in_cells = {
      std::array<DataVector, 2>{{zero_dv, zero_dv}},
      std::array<DataVector, 2>{{zero_dv, zero_dv}}};
  std::array<DataVector, 2> single_packet_energy = {zero_dv, zero_dv};
  const std::array<double, 2> energy_at_bin_center = {2.0, 5.0};
  InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>
      inverse_jacobian = make_with_value<
          InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>>(
          zero_dv, 0.0);
  Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid> jacobian =
      make_with_value<Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>>(
          zero_dv, 0.0);

  // Set data
  for (size_t s = 0; s < 2; s++) {
    for (size_t i = 0; i < zero_dv.size(); i++) {
      gsl::at(single_packet_energy, s)[i] = 2.0;
      for (size_t g = 0; g < 2; g++) {
        gsl::at(gsl::at(emission_in_cells, s), g)[i] =
            static_cast<double>(2 - 2 * s);
      }
    }
  }

  // Set Jacobian to a Lorentz boost in the y-direction with W=2
  const double v_boost = sqrt(3) / 2.0;
  const double W_boost = 2.0;

  jacobian.get(0, 0) = W_boost;
  jacobian.get(0, 2) = -W_boost * v_boost;
  jacobian.get(1, 1) = 1.0;
  jacobian.get(2, 0) = -W_boost * v_boost;
  jacobian.get(2, 2) = W_boost;
  jacobian.get(3, 3) = 1.0;
  inverse_jacobian.get(0, 0) = W_boost;
  inverse_jacobian.get(0, 2) = W_boost * v_boost;
  inverse_jacobian.get(1, 1) = 1.0;
  inverse_jacobian.get(2, 0) = W_boost * v_boost;
  inverse_jacobian.get(2, 2) = W_boost;
  inverse_jacobian.get(3, 3) = 1.0;

  tnsr::i<DataVector, 3, Frame::Inertial> lower_spatial_four_velocity =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(zero_dv, 0.0);
  lower_spatial_four_velocity.get(1) = W_boost * v_boost;
  Scalar<DataVector> lorentz_factor =
      make_with_value<Scalar<DataVector>>(zero_dv, W_boost);

  Scalar<DataVector> coupling_tilde_tau =
      make_with_value<Scalar<DataVector>>(zero_dv, 0.0);
  Scalar<DataVector> coupling_rho_ye =
      make_with_value<Scalar<DataVector>>(zero_dv, 0.0);
  tnsr::i<DataVector, 3, Frame::Inertial> coupling_tilde_s =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(zero_dv, 0.0);

  // Run emission code
  Particles::MonteCarlo::TemplatedLocalFunctions<2, 2> MonteCarloStruct;

  MonteCarloStruct.emit_packets(
      &all_packets, &generator, &coupling_tilde_tau, &coupling_tilde_s,
      &coupling_rho_ye, time, time_step, mesh, emission_in_cells,
      single_packet_energy, energy_at_bin_center, lorentz_factor,
      lower_spatial_four_velocity, jacobian, inverse_jacobian);

  // Check some things
  const size_t n_packets = all_packets.size();
  CHECK(n_packets == 54);
  for (size_t n = 0; n < n_packets; n++) {
    // Check that -p_mu u^\mu is the expected energy of the neutrinos
    const double neutrino_energy =
        W_boost * (all_packets[n].momentum_upper_t -
                   all_packets[n].momentum.get(1) * v_boost);
    CHECK(fabs(all_packets[n].number_of_neutrinos * neutrino_energy - 2.0) <
          1.e-14);
  }
  CHECK(gsl::at(energy_at_bin_center, 0) == 2.0);

  Scalar<DataVector> expected_coupling_tilde_tau =
      make_with_value<Scalar<DataVector>>(zero_dv, -8.0);
  Scalar<DataVector> expected_coupling_rho_ye =
      make_with_value<Scalar<DataVector>>(zero_dv, -1.4 * proton_mass);
  tnsr::i<DataVector, 3, Frame::Inertial> expected_coupling_tilde_s =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(zero_dv, 0.0);
  expected_coupling_tilde_s.get(1) = -4.0 * sqrt(3.0);

  CHECK_ITERABLE_CUSTOM_APPROX(
      coupling_tilde_tau, expected_coupling_tilde_tau,
      Approx::custom().epsilon(epsilon_approx).scale(1.0));
  CHECK_ITERABLE_CUSTOM_APPROX(
      coupling_rho_ye, expected_coupling_rho_ye,
      Approx::custom().epsilon(epsilon_approx).scale(1.0));
  CHECK_ITERABLE_CUSTOM_APPROX(
      coupling_tilde_s, expected_coupling_tilde_s,
      Approx::custom().epsilon(epsilon_approx).scale(1.0));
}
