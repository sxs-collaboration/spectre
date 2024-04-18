// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <random>

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Particles/MonteCarlo/CouplingTermsForPropagation.hpp"
#include "Evolution/Particles/MonteCarlo/EvolvePackets.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Evolution/Particles/MonteCarlo/Scattering.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Frame {
struct ElementLogical;
struct Fluid;
struct Inertial;
}  // namespace Frame

namespace {

void test_single_scatter() {
  MAKE_GENERATOR(generator);

  DataVector zero_dv(8);
  zero_dv = 0.0;
  InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>
      inverse_jacobian = make_with_value<
          InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>>(
          zero_dv, 0.0);
  Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid> jacobian =
      make_with_value<Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>>(
          zero_dv, 0.0);

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

  Particles::MonteCarlo::Packet packet(0, 1.0, 0, 0.0, -1.0, -1.0, -1.0, 1.0,
                                       1.0, 0.0, 0.0);

  auto boosted_frame_energy = [&W_boost, &v_boost, &packet]() {
    return W_boost * (packet.momentum_upper_t - packet.momentum[1] * v_boost);
  };

  const double initial_fluid_frame_energy = boosted_frame_energy();
  for (int step = 0; step < 5; step++) {
    scatter_packet(&packet, &generator, initial_fluid_frame_energy, jacobian,
                   inverse_jacobian);
    const double current_fluid_frame_energy = boosted_frame_energy();
    CHECK(fabs(initial_fluid_frame_energy - current_fluid_frame_energy) <
          1.e-14);
  }
}

void test_diffusion_params() {
  const Particles::MonteCarlo::DiffusionMonteCarloParameters diffusion_params;
  const double epsilon_integral = 1.e-4;

  // Numbers checked through independent numerical integration
  // Check that we recover Eq 29 of 10.1093/mnras/sty108 at 5 points
  CHECK(fabs(gsl::at(diffusion_params.ScatteringRofP, 0) - 0.0) <
        epsilon_integral);
  CHECK(fabs(gsl::at(diffusion_params.ScatteringRofP, 250) - 0.778631) <
        epsilon_integral);
  CHECK(fabs(gsl::at(diffusion_params.ScatteringRofP, 500) - 1.08765) <
        epsilon_integral);
  CHECK(fabs(gsl::at(diffusion_params.ScatteringRofP, 750) - 1.43324) <
        epsilon_integral);
  CHECK(fabs(gsl::at(diffusion_params.ScatteringRofP, 999) - 2.85186) <
        epsilon_integral);
  // Check that we recover Eq. 30 of 10.1093/mnras/sty108 at 4 points
  CHECK(fabs(gsl::at(diffusion_params.OpacityDependentCorrectionToRofP, 0) -
             0.828982) < epsilon_integral);
  CHECK(fabs(gsl::at(diffusion_params.OpacityDependentCorrectionToRofP, 25) -
             0.948835) < epsilon_integral);
  CHECK(fabs(gsl::at(diffusion_params.OpacityDependentCorrectionToRofP, 50) -
             0.986138) < epsilon_integral);
  CHECK(fabs(gsl::at(diffusion_params.OpacityDependentCorrectionToRofP, 75) -
             0.996452) < epsilon_integral);
  CHECK(fabs(gsl::at(diffusion_params.OpacityDependentCorrectionToRofP, 100) -
             0.999121) < epsilon_integral);
}

void test_diffusion() {
  const Particles::MonteCarlo::DiffusionMonteCarloParameters diffusion_params;

  MAKE_GENERATOR(generator);
  const double time_step = 0.5;
  const double scattering_opacity = 50.0;
  const double absorption_opacity = 1.e-60;

  const size_t dv_size = 1;
  DataVector zero_dv(dv_size);
  zero_dv = 0.0;
  Scalar<DataVector> lapse(dv_size, 1.0);
  Scalar<DataVector> lorentz_factor(dv_size, 1.0);
  tnsr::i<DataVector, 3, Frame::Inertial> lower_spatial_four_velocity =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  tnsr::I<DataVector, 3, Frame::Inertial> shift =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  inv_spatial_metric.get(0, 0) = 1.0;
  inv_spatial_metric.get(1, 1) = 1.0;
  inv_spatial_metric.get(2, 2) = 1.0;
  spatial_metric.get(0, 0) = 1.0;
  spatial_metric.get(1, 1) = 1.0;
  spatial_metric.get(2, 2) = 1.0;
  tnsr::i<DataVector, 3, Frame::Inertial> d_lapse =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  tnsr::iJ<DataVector, 3, Frame::Inertial> d_shift =
      make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  tnsr::iJJ<DataVector, 3, Frame::Inertial> d_inv_spatial_metric =
      make_with_value<tnsr::iJJ<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> mesh_velocity =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  mesh_velocity.value().get(0) = 0.1;
  mesh_velocity.value().get(1) = 0.2;
  mesh_velocity.value().get(2) = 0.3;
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
      inverse_jacobian_logical_to_inertial =
          make_with_value<InverseJacobian<DataVector, 3, Frame::ElementLogical,
                                          Frame::Inertial>>(lapse, 0.0);
  inverse_jacobian_logical_to_inertial.get(0, 0) = 1.0;
  inverse_jacobian_logical_to_inertial.get(1, 1) = 1.0;
  inverse_jacobian_logical_to_inertial.get(2, 2) = 1.0;

  Scalar<DataVector> coupling_tilde_tau =
      make_with_value<Scalar<DataVector>>(zero_dv, 0.0);
  Scalar<DataVector> coupling_rho_ye =
      make_with_value<Scalar<DataVector>>(zero_dv, 0.0);
  tnsr::i<DataVector, 3, Frame::Inertial> coupling_tilde_s =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(zero_dv, 0.0);

  DataVector prefactor_diffusion_time_vector = zero_dv;
  DataVector prefactor_diffusion_four_velocity = zero_dv;
  DataVector prefactor_diffusion_time_step = zero_dv;
  Particles::MonteCarlo::DiffusionPrecomputeForElement(
      &prefactor_diffusion_time_vector, &prefactor_diffusion_four_velocity,
      &prefactor_diffusion_time_step, lorentz_factor,
      lower_spatial_four_velocity, lapse, shift, spatial_metric);

  for (size_t i = 0; i < prefactor_diffusion_time_vector.size(); i++) {
    CHECK(std::isnan(prefactor_diffusion_time_vector[i]));
    CHECK(std::isnan(prefactor_diffusion_four_velocity[i]));
  }
  CHECK(prefactor_diffusion_time_step == -1.0);

  Parallel::printf("Diffusion limit:\n");
  for (size_t i = 0; i < 100000; i++) {
    Particles::MonteCarlo::Packet packet(0, 1.0, 0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                         1.0, 0.0, 0.0);
    double neutrino_energy = 1.0;
    Particles::MonteCarlo::diffuse_packet(
        &packet, &generator, &neutrino_energy, &coupling_tilde_tau,
        &coupling_tilde_s, &coupling_rho_ye, time_step, diffusion_params,
        absorption_opacity, scattering_opacity, lorentz_factor,
        lower_spatial_four_velocity, lapse, shift, d_lapse, d_shift,
        d_inv_spatial_metric, spatial_metric, inv_spatial_metric, mesh_velocity,
        inverse_jacobian_logical_to_inertial, jacobian_inertial_to_fluid,
        inverse_jacobian_inertial_to_fluid, prefactor_diffusion_time_step,
        prefactor_diffusion_four_velocity, prefactor_diffusion_time_vector);

    if (mesh_velocity.has_value()) {
      for (size_t d = 0; d < 3; d++) {
        packet.coordinates[d] += mesh_velocity.value().get(d)[0] * time_step;
      }
    }

    const double rad = sqrt(packet.coordinates[0] * packet.coordinates[0] +
                            packet.coordinates[1] * packet.coordinates[1] +
                            packet.coordinates[2] * packet.coordinates[2]);
    const double cth = packet.coordinates[2] / rad;
    const double phi = atan2(packet.coordinates[1], packet.coordinates[0]);
    const double pitch = (packet.coordinates[0] * packet.momentum[0] +
                          packet.coordinates[1] * packet.momentum[1] +
                          packet.coordinates[2] * packet.momentum[2]) /
                         neutrino_energy / rad;
    Parallel::printf("%.5f %.5f %.5f %.5f 0\n", rad, cth, phi, pitch);
  }

  Parallel::printf("Coupling: %.5e %.5e %.5e %.5e %.5e\n",
                   get(coupling_tilde_tau)[0], coupling_tilde_s.get(0)[0],
                   coupling_tilde_s.get(1)[0], coupling_tilde_s.get(2)[0],
                   get(coupling_rho_ye)[0]);
  // Reset coupling terms
  get(coupling_tilde_tau)[0] = 0.0;
  for (size_t d = 0; d < 3; d++) {
    coupling_tilde_s.get(d)[0] = 0.0;
  }
  get(coupling_rho_ye)[0] = 0.0;

  Parallel::printf("Full scattering:\n");
  std::uniform_real_distribution<double> rng_uniform_zero_to_one(0.0, 1.0);
  double neutrino_energy = 1.0;
  double dt_scattering = 0.0;
  for (size_t i = 0; i < 100000; i++) {
    Particles::MonteCarlo::Packet packet(0, 1.0, 0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                         1.0, 0.0, 0.0);
    while (packet.time < time_step) {
      neutrino_energy = Particles::MonteCarlo::compute_fluid_frame_energy(
          packet, lorentz_factor, lower_spatial_four_velocity, lapse,
          inv_spatial_metric);
      Particles::MonteCarlo::scatter_packet(
          &packet, &generator, neutrino_energy, jacobian_inertial_to_fluid,
          inverse_jacobian_inertial_to_fluid);
      dt_scattering = std::clamp(
          -log(rng_uniform_zero_to_one(generator) + 1.e-70) /
              (scattering_opacity)*packet.momentum_upper_t / neutrino_energy,
          0.0, time_step - packet.time);

      Particles::MonteCarlo::evolve_single_packet_on_geodesic(
          &packet, dt_scattering, lapse, shift, d_lapse, d_shift,
          d_inv_spatial_metric, inv_spatial_metric, mesh_velocity,
          inverse_jacobian_logical_to_inertial);
      const std::array<double, 3> lower_spatial_four_velocity_packet = {
          lower_spatial_four_velocity.get(0)[0],
          lower_spatial_four_velocity.get(1)[0],
          lower_spatial_four_velocity.get(2)[0]};
      Particles::MonteCarlo::AddCouplingTermsForPropagation
        (&coupling_tilde_tau, &coupling_tilde_s,
         &coupling_rho_ye, packet, dt_scattering,
         absorption_opacity, 0.0, neutrino_energy,
         get(lapse)[0], get(lorentz_factor)[0],
         lower_spatial_four_velocity_packet);
    }

    if (mesh_velocity.has_value()) {
      for (size_t d = 0; d < 3; d++) {
        packet.coordinates[d] += mesh_velocity.value().get(d)[0] * time_step;
      }
    }

    const double rad = sqrt(packet.coordinates[0] * packet.coordinates[0] +
                            packet.coordinates[1] * packet.coordinates[1] +
                            packet.coordinates[2] * packet.coordinates[2]);
    const double cth = packet.coordinates[2] / rad;
    const double phi = atan2(packet.coordinates[1], packet.coordinates[0]);
    const double pitch = (packet.coordinates[0] * packet.momentum[0] +
                          packet.coordinates[1] * packet.momentum[1] +
                          packet.coordinates[2] * packet.momentum[2]) /
                         neutrino_energy / rad;
    Parallel::printf("%.5f %.5f %.5f %.5f 1\n", rad, cth, phi, pitch);
  }
  Parallel::printf("Coupling: %.5e %.5e %.5e %.5e %.5e\n",
                   get(coupling_tilde_tau)[0], coupling_tilde_s.get(0)[0],
                   coupling_tilde_s.get(1)[0], coupling_tilde_s.get(2)[0],
                   get(coupling_rho_ye)[0]);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Particles.MonteCarloScattering",
                  "[Unit][Evolution]") {
  test_single_scatter();
  test_diffusion_params();
  // Not turned on by defaults... too long for automated tests,
  // but useful framework to test diffusion regime.
  // NOLINTNEXTLINE(readability-simplify-boolean-expr)
  if ((false)) {
    test_diffusion();
  }
}
