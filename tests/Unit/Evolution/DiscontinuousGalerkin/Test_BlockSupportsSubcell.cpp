// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "Domain/Block.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Evolution/DiscontinuousGalerkin/BlockSupportsSubcell.hpp"

namespace {
template <size_t Dim>
Block<Dim> make_block(const std::array<domain::Topology, Dim>& topologies) {
  return Block<Dim>{nullptr, 0, DirectionMap<Dim, BlockNeighbors<Dim>>{},
                    "Block", topologies};
}

void test_1d() {
  CHECK(evolution::dg::block_supports_subcell(
      make_block<1>(domain::topologies::hypercube<1>)));
  CHECK(evolution::dg::block_supports_subcell(
      make_block<1>(std::array{domain::Topology::B1Radial})));
  CHECK_FALSE(evolution::dg::block_supports_subcell(
      make_block<1>(std::array{domain::Topology::S1})));
}

void test_2d() {
  // Rectangles support subcell
  CHECK(evolution::dg::block_supports_subcell(
      make_block<2>(domain::topologies::hypercube<2>)));
  // Annulus does not support subcell
  CHECK_FALSE(evolution::dg::block_supports_subcell(
      make_block<2>(domain::topologies::annulus)));
  // Disk does not support subcell
  CHECK_FALSE(evolution::dg::block_supports_subcell(
      make_block<2>(domain::topologies::disk)));
}

void test_3d() {
  // Cubes support subcell
  CHECK(evolution::dg::block_supports_subcell(
      make_block<3>(domain::topologies::hypercube<3>)));
  // Cartoon topologies support subcell
  CHECK(evolution::dg::block_supports_subcell(
      make_block<3>(domain::topologies::cartoon_sphere)));
  CHECK(evolution::dg::block_supports_subcell(
      make_block<3>(domain::topologies::cartoon_sphere_inner)));
  CHECK(evolution::dg::block_supports_subcell(
      make_block<3>(domain::topologies::cartoon_cylinder)));
  CHECK(evolution::dg::block_supports_subcell(
      make_block<3>(domain::topologies::cartoon_cylinder_inner)));
  // Spherical shells do not support subcell
  CHECK_FALSE(evolution::dg::block_supports_subcell(
      make_block<3>(domain::topologies::spherical_shell)));
  // Filled spheres do not support subcell
  CHECK_FALSE(evolution::dg::block_supports_subcell(
      make_block<3>(domain::topologies::full_sphere)));
  // Full cylinders do not support subcell
  CHECK_FALSE(evolution::dg::block_supports_subcell(
      make_block<3>(domain::topologies::full_cylinder)));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.BlockSupportsSubcell",
                  "[Unit][Evolution]") {
  test_1d();
  test_2d();
  test_3d();
}
