// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Particles/MonteCarlo/EvolvePackets.hpp"
#include "Evolution/Particles/MonteCarlo/NeutrinoInteractionTable.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Evolution/Particles/MonteCarlo/TemplatedLocalFunctions.hpp"
#include "Framework/TestHelpers.hpp"
#include "Framework/TestingFramework.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace{

void test_flat_space_time_step() {
  const Mesh<3> mesh(2, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered);

  MAKE_GENERATOR(generator);

  const size_t NeutrinoSpecies = 3;
  const size_t NeutrinoEnergies = 4;

  const size_t dv_size = 8;
  DataVector zero_dv(dv_size, 0.0);

  // Minkowski metric
  Scalar<DataVector> lapse{DataVector(dv_size, 1.0)};
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
  Scalar<DataVector> determinant_spatial_metric(dv_size, 1.0);
  tnsr::I<DataVector, 3, Frame::Inertial> shift =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
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
      inverse_jacobian_logical_to_inertial =
          make_with_value<InverseJacobian<DataVector, 3, Frame::ElementLogical,
                                          Frame::Inertial>>(lapse, 0.0);
  inverse_jacobian_logical_to_inertial.get(0, 0) = 1.0;
  inverse_jacobian_logical_to_inertial.get(1, 1) = 1.0;
  inverse_jacobian_logical_to_inertial.get(2, 2) = 1.0;
  Scalar<DataVector> det_jacobian_logical_to_inertial(dv_size, 1.0);

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

  Particles::MonteCarlo::Packet packet(1, 1.0, 0, 0.0, -1.0, -1.0, -1.0, 1.0,
                                       1.0, 0.0, 0.0);
  packet.renormalize_momentum(inv_spatial_metric, lapse);
  CHECK(packet.momentum_upper_t == 1.0);

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
  Particles::MonteCarlo::TemplatedLocalFunctions<NeutrinoEnergies,
                                                 NeutrinoSpecies>
      MonteCarloStruct;

  const double start_time = 0.0;
  const double final_time = 0.2;
  std::array<DataVector, NeutrinoSpecies> single_packet_energy = {
      zero_dv, zero_dv, zero_dv};
  for (size_t s = 0; s < NeutrinoSpecies; s++) {
    gsl::at(single_packet_energy,s) = 1.0;
  }
  const std::string h5_file_name{
      unit_test_src_path() +
      "Evolution/Particles/MonteCarlo/NuLib_TestTable.h5"};
  const Particles::MonteCarlo::NeutrinoInteractionTable<4, 3> interaction_table(
      h5_file_name);

  Scalar<DataVector> baryon_density(dv_size, 1.619109365278362e-05);
  Scalar<DataVector> temperature(dv_size, 10.0);
  Scalar<DataVector> electron_fraction(dv_size, 0.05);
  Scalar<DataVector> lorentz_factor(dv_size, 1.0);
  tnsr::i<DataVector, 3, Frame::Inertial> lower_spatial_four_velocity =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(lapse, 0.0);

  MonteCarloStruct.take_time_step_on_element(
      &packets, &generator, &single_packet_energy, start_time, final_time,
      interaction_table, electron_fraction, baryon_density, temperature,
      lorentz_factor, lower_spatial_four_velocity, lapse, shift, d_lapse,
      d_shift, d_inv_spatial_metric, spatial_metric, inv_spatial_metric,
      determinant_spatial_metric, mesh, mesh_coordinates, inertial_coordinates,
      mesh_velocity,
      inverse_jacobian_logical_to_inertial, det_jacobian_logical_to_inertial,
      jacobian_inertial_to_fluid, inverse_jacobian_inertial_to_fluid);

  size_t n_packets = packets.size();
  // In the current setup, we just propagate a single packet to the
  // final time.
  CHECK(n_packets==1);
  CHECK(packets[0].time==0.2);
}

} // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Particles.MonteCarloTakeTimeStep",
                  "[Unit][Evolution]") {
  test_flat_space_time_step();
}
