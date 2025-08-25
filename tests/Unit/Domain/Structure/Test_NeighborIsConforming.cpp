// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/NeighborIsConforming.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Topology.hpp"

namespace {
void test_1d() {
  for (const auto& direction : Direction<1>::all_directions()) {
    for (const auto& orientation :
         std::array{OrientationMap<1>::create_aligned(),
                    OrientationMap<1>(std::array{Direction<1>::lower_xi()})}) {
      CHECK(neighbor_is_conforming(std::array{domain::Topology::I1},
                                   std::array{domain::Topology::I1}, direction,
                                   orientation));
    }
  }
}

void test_2d() {
  const OrientationMap<2> aligned = OrientationMap<2>::create_aligned();
  const OrientationMap<2> quarter_turn_ccw(std::array<Direction<2>, 2>{
      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}});
  const OrientationMap<2> half_turn(std::array<Direction<2>, 2>{
      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}});
  const OrientationMap<2> quarter_turn_cw(std::array<Direction<2>, 2>{
      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}});

  for (const auto& direction : Direction<2>::all_directions()) {
    for (const auto& orientation :
         std::array{aligned, quarter_turn_ccw, half_turn, quarter_turn_cw}) {
      CHECK(neighbor_is_conforming(domain::topologies::hypercube<2>,
                                   domain::topologies::hypercube<2>, direction,
                                   orientation));
    }
  }

  CHECK(neighbor_is_conforming(domain::topologies::disk,
                               domain::topologies::annulus,
                               Direction<2>::upper_xi(), aligned));
  CHECK(neighbor_is_conforming(domain::topologies::annulus,
                               domain::topologies::disk,
                               Direction<2>::lower_xi(), aligned));
  CHECK(neighbor_is_conforming(domain::topologies::annulus,
                               domain::topologies::annulus,
                               Direction<2>::lower_xi(), aligned));
  CHECK(neighbor_is_conforming(domain::topologies::annulus,
                               domain::topologies::annulus,
                               Direction<2>::upper_xi(), aligned));

  for (const auto& orientation :
       std::array{aligned, quarter_turn_ccw, half_turn, quarter_turn_cw}) {
    CHECK_FALSE(neighbor_is_conforming(domain::topologies::disk,
                                       domain::topologies::hypercube<2>,
                                       Direction<2>::upper_xi(), orientation));
    CHECK_FALSE(neighbor_is_conforming(domain::topologies::annulus,
                                       domain::topologies::hypercube<2>,
                                       Direction<2>::lower_xi(), orientation));
    CHECK_FALSE(neighbor_is_conforming(domain::topologies::annulus,
                                       domain::topologies::hypercube<2>,
                                       Direction<2>::upper_xi(), orientation));
  }
}

void test_3d() {
  const OrientationMap<3> aligned = OrientationMap<3>::create_aligned();
  CHECK(neighbor_is_conforming(domain::topologies::spherical_shell,
                               domain::topologies::spherical_shell,
                               Direction<3>::lower_xi(), aligned));
  CHECK(neighbor_is_conforming(domain::topologies::spherical_shell,
                               domain::topologies::full_sphere,
                               Direction<3>::lower_xi(), aligned));
  CHECK(neighbor_is_conforming(domain::topologies::spherical_shell,
                               domain::topologies::spherical_shell,
                               Direction<3>::upper_xi(), aligned));
  CHECK(neighbor_is_conforming(domain::topologies::full_sphere,
                               domain::topologies::spherical_shell,
                               Direction<3>::upper_xi(), aligned));
  CHECK(neighbor_is_conforming(domain::topologies::full_cylinder,
                               domain::topologies::full_cylinder,
                               Direction<3>::lower_zeta(), aligned));
  CHECK(neighbor_is_conforming(domain::topologies::full_cylinder,
                               domain::topologies::full_cylinder,
                               Direction<3>::upper_zeta(), aligned));
  CHECK(neighbor_is_conforming(domain::topologies::cylindrical_shell,
                               domain::topologies::cylindrical_shell,
                               Direction<3>::lower_xi(), aligned));
  CHECK(neighbor_is_conforming(domain::topologies::cylindrical_shell,
                               domain::topologies::cylindrical_shell,
                               Direction<3>::upper_xi(), aligned));
  CHECK(neighbor_is_conforming(domain::topologies::cylindrical_shell,
                               domain::topologies::cylindrical_shell,
                               Direction<3>::lower_zeta(), aligned));
  CHECK(neighbor_is_conforming(domain::topologies::cylindrical_shell,
                               domain::topologies::cylindrical_shell,
                               Direction<3>::upper_zeta(), aligned));

  const OrientationMap<3> radial_xi_to_zeta(std::array<Direction<3>, 3>{
      Direction<3>::upper_zeta(), Direction<3>::self(), Direction<3>::self()});
  CHECK_FALSE(neighbor_is_conforming(
      domain::topologies::spherical_shell, domain::topologies::hypercube<3>,
      Direction<3>::lower_xi(), radial_xi_to_zeta));
  CHECK_FALSE(neighbor_is_conforming(
      domain::topologies::spherical_shell, domain::topologies::hypercube<3>,
      Direction<3>::upper_xi(), radial_xi_to_zeta));
  CHECK_FALSE(neighbor_is_conforming(
      domain::topologies::full_sphere, domain::topologies::hypercube<3>,
      Direction<3>::upper_xi(), radial_xi_to_zeta));

  CHECK_FALSE(neighbor_is_conforming(
      domain::topologies::spherical_shell, domain::topologies::full_cylinder,
      Direction<3>::lower_xi(), radial_xi_to_zeta));
  CHECK_FALSE(neighbor_is_conforming(
      domain::topologies::spherical_shell, domain::topologies::full_cylinder,
      Direction<3>::upper_xi(), radial_xi_to_zeta));
  CHECK_FALSE(neighbor_is_conforming(
      domain::topologies::full_sphere, domain::topologies::full_cylinder,
      Direction<3>::upper_xi(), radial_xi_to_zeta));

  const OrientationMap<3> radial_aligned(std::array<Direction<3>, 3>{
      Direction<3>::upper_xi(), Direction<3>::self(), Direction<3>::self()});
  CHECK_FALSE(neighbor_is_conforming(domain::topologies::spherical_shell,
                                     domain::topologies::cylindrical_shell,
                                     Direction<3>::lower_xi(), radial_aligned));
  CHECK_FALSE(neighbor_is_conforming(domain::topologies::spherical_shell,
                                     domain::topologies::cylindrical_shell,
                                     Direction<3>::upper_xi(), radial_aligned));
  CHECK_FALSE(neighbor_is_conforming(domain::topologies::full_sphere,
                                     domain::topologies::cylindrical_shell,
                                     Direction<3>::upper_xi(), radial_aligned));
  CHECK_FALSE(neighbor_is_conforming(domain::topologies::spherical_shell,
                                     domain::topologies::cylindrical_shell,
                                     Direction<3>::lower_xi(),
                                     radial_xi_to_zeta));
  CHECK_FALSE(neighbor_is_conforming(domain::topologies::spherical_shell,
                                     domain::topologies::cylindrical_shell,
                                     Direction<3>::upper_xi(),
                                     radial_xi_to_zeta));
  CHECK_FALSE(neighbor_is_conforming(
      domain::topologies::full_sphere, domain::topologies::cylindrical_shell,
      Direction<3>::upper_xi(), radial_xi_to_zeta));

  CHECK_FALSE(neighbor_is_conforming(domain::topologies::full_cylinder,
                                     domain::topologies::hypercube<3>,
                                     Direction<3>::upper_xi(), aligned));
  CHECK_FALSE(neighbor_is_conforming(domain::topologies::full_cylinder,
                                     domain::topologies::hypercube<3>,
                                     Direction<3>::lower_zeta(), aligned));
  CHECK_FALSE(neighbor_is_conforming(domain::topologies::full_cylinder,
                                     domain::topologies::hypercube<3>,
                                     Direction<3>::upper_zeta(), aligned));
  const OrientationMap<3> radial_zeta_to_xi(std::array<Direction<3>, 3>{
      Direction<3>::self(), Direction<3>::self(), Direction<3>::upper_xi()});
  CHECK_FALSE(neighbor_is_conforming(
      domain::topologies::full_cylinder, domain::topologies::spherical_shell,
      Direction<3>::lower_zeta(), radial_zeta_to_xi));
  CHECK_FALSE(neighbor_is_conforming(
      domain::topologies::full_cylinder, domain::topologies::spherical_shell,
      Direction<3>::upper_zeta(), radial_zeta_to_xi));
  CHECK_FALSE(neighbor_is_conforming(
      domain::topologies::full_cylinder, domain::topologies::full_sphere,
      Direction<3>::lower_zeta(), radial_zeta_to_xi));
  CHECK_FALSE(neighbor_is_conforming(
      domain::topologies::full_cylinder, domain::topologies::full_sphere,
      Direction<3>::upper_zeta(), radial_zeta_to_xi));
  CHECK(neighbor_is_conforming(domain::topologies::full_cylinder,
                               domain::topologies::cylindrical_shell,
                               Direction<3>::upper_xi(), aligned));
  const OrientationMap<3> full_to_shell(std::array<Direction<3>, 3>{
      Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
      Direction<3>::upper_xi()});
  CHECK(neighbor_is_conforming(domain::topologies::full_cylinder,
                               domain::topologies::cylindrical_shell,
                               Direction<3>::upper_xi(), full_to_shell));

  CHECK_FALSE(neighbor_is_conforming(domain::topologies::cylindrical_shell,
                                     domain::topologies::hypercube<3>,
                                     Direction<3>::lower_xi(), aligned));
  CHECK_FALSE(neighbor_is_conforming(domain::topologies::cylindrical_shell,
                                     domain::topologies::hypercube<3>,
                                     Direction<3>::upper_xi(), aligned));
  CHECK_FALSE(neighbor_is_conforming(domain::topologies::cylindrical_shell,
                                     domain::topologies::hypercube<3>,
                                     Direction<3>::lower_zeta(), aligned));
  CHECK_FALSE(neighbor_is_conforming(domain::topologies::cylindrical_shell,
                                     domain::topologies::hypercube<3>,
                                     Direction<3>::upper_zeta(), aligned));
  CHECK_FALSE(neighbor_is_conforming(domain::topologies::cylindrical_shell,
                                     domain::topologies::spherical_shell,
                                     Direction<3>::lower_xi(), radial_aligned));
  CHECK_FALSE(neighbor_is_conforming(domain::topologies::cylindrical_shell,
                                     domain::topologies::spherical_shell,
                                     Direction<3>::upper_xi(), radial_aligned));
  CHECK_FALSE(neighbor_is_conforming(domain::topologies::cylindrical_shell,
                                     domain::topologies::spherical_shell,
                                     Direction<3>::lower_zeta(),
                                     radial_zeta_to_xi));
  CHECK_FALSE(neighbor_is_conforming(domain::topologies::cylindrical_shell,
                                     domain::topologies::spherical_shell,
                                     Direction<3>::upper_zeta(),
                                     radial_zeta_to_xi));
  CHECK_FALSE(neighbor_is_conforming(domain::topologies::cylindrical_shell,
                                     domain::topologies::full_sphere,
                                     Direction<3>::lower_xi(), radial_aligned));
  CHECK_FALSE(neighbor_is_conforming(domain::topologies::cylindrical_shell,
                                     domain::topologies::full_sphere,
                                     Direction<3>::upper_xi(), radial_aligned));
  CHECK_FALSE(neighbor_is_conforming(
      domain::topologies::cylindrical_shell, domain::topologies::full_sphere,
      Direction<3>::lower_zeta(), radial_zeta_to_xi));
  CHECK_FALSE(neighbor_is_conforming(
      domain::topologies::cylindrical_shell, domain::topologies::full_sphere,
      Direction<3>::upper_zeta(), radial_zeta_to_xi));
  CHECK(neighbor_is_conforming(domain::topologies::cylindrical_shell,
                               domain::topologies::full_cylinder,
                               Direction<3>::lower_xi(), aligned));
  const OrientationMap<3> shell_to_full(std::array<Direction<3>, 3>{
      Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
      Direction<3>::lower_xi()});
  CHECK(neighbor_is_conforming(domain::topologies::cylindrical_shell,
                               domain::topologies::full_cylinder,
                               Direction<3>::upper_zeta(), shell_to_full));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Structure.NeighborIsConforming",
                  "[Domain][Unit]") {
  test_1d();
  test_2d();
  test_3d();
}
