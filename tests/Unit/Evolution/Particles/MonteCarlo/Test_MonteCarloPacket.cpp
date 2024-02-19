// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Particles/MonteCarlo/EvolveMonteCarloPackets.hpp"
#include "Evolution/Particles/MonteCarlo/MonteCarloPacket.hpp"
#include "Framework/TestingFramework.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Particles.MonteCarlo", "[Unit][Evolution]") {
  Particles::MonteCarlo::Packet packet(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0);

  const Mesh<3> mesh(2, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered);

  // Minkowski metric
  Scalar<DataVector> lapse{DataVector{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};

  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  inv_spatial_metric.get(0, 0) = 1.0;
  inv_spatial_metric.get(1, 1) = 1.0;
  inv_spatial_metric.get(2, 2) = 1.0;

  tnsr::I<DataVector, 3, Frame::Inertial> shift =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  tnsr::i<DataVector, 3, Frame::Inertial> d_lapse =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  tnsr::iJ<DataVector, 3, Frame::Inertial> d_shift =
      make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  tnsr::iJJ<DataVector, 3, Frame::Inertial> d_inv_spatial_metric =
      make_with_value<tnsr::iJJ<DataVector, 3, Frame::Inertial>>(lapse, 0.0);

  // Coordinates
  tnsr::I<DataVector, 3, Frame::Inertial> mesh_coordinates =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  mesh_coordinates.get(0) = DataVector{0., 1., 0., 1., 0., 1., 0., 1.};
  mesh_coordinates.get(1) = DataVector{0., 0., 1., 1., 0., 0., 1., 1.};
  mesh_coordinates.get(2) = DataVector{0., 0., 0., 0., 1., 1., 1., 1.};

  const size_t closest_point_index = 0;

  packet.renormalize_momentum(inv_spatial_metric, lapse, closest_point_index);
  CHECK(packet.momentum_upper_t == 1.0);

  std::vector<Particles::MonteCarlo::Packet> packets{packet};
  Particles::MonteCarlo::evolve_packets(
      &packets, 1.0, mesh, mesh_coordinates, lapse, shift, d_lapse, d_shift,
      d_inv_spatial_metric, inv_spatial_metric);
  CHECK(packets[0].coordinates.get(0) == 1.0);
  CHECK(packets[0].coordinates.get(1) == 0.0);
  CHECK(packets[0].coordinates.get(2) == 0.0);
  CHECK(packets[0].momentum.get(0) == 1.0);
  CHECK(packets[0].momentum.get(1) == 0.0);
  CHECK(packets[0].momentum.get(2) == 0.0);
  CHECK(packets[0].time == 1.0);
}
