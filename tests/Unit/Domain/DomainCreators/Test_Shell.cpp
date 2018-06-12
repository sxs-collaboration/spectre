// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <unordered_set>
#include <vector>

#include "Domain/BlockNeighbor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Wedge3D.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainCreators/DomainCreator.hpp"
#include "Domain/DomainCreators/Shell.hpp"
#include "Domain/OrientationMap.hpp"
#include "Utilities/MakeArray.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/TestCreation.hpp"

/// \cond
namespace Frame {
struct Inertial;
struct Logical;
}  // namespace Frame
/// \endcond

namespace {
void test_shell_construction(
    const DomainCreators::Shell<Frame::Inertial>& shell,
    const double inner_radius, const double outer_radius,
    const bool use_equiangular_map,
    const std::array<size_t, 2>& expected_shell_extents,
    const std::vector<std::array<size_t, 3>>& expected_refinement_level) {
  const auto domain = shell.create_domain();
  const OrientationMap<3> aligned_orientation{};
  const OrientationMap<3> quarter_turn_ccw_about_zeta(
      std::array<Direction<3>, 3>{{Direction<3>::lower_eta(),
                                   Direction<3>::upper_xi(),
                                   Direction<3>::upper_zeta()}});
  const OrientationMap<3> half_turn_about_zeta(std::array<Direction<3>, 3>{
      {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
       Direction<3>::upper_zeta()}});
  const OrientationMap<3> quarter_turn_cw_about_zeta(
      std::array<Direction<3>, 3>{{Direction<3>::upper_eta(),
                                   Direction<3>::lower_xi(),
                                   Direction<3>::upper_zeta()}});
  const std::vector<DirectionMap<3, BlockNeighbor<3>>> expected_block_neighbors{
      {{Direction<3>::upper_xi(), {4, quarter_turn_ccw_about_zeta}},
       {Direction<3>::upper_eta(), {2, aligned_orientation}},
       {Direction<3>::lower_xi(), {5, quarter_turn_cw_about_zeta}},
       {Direction<3>::lower_eta(), {3, aligned_orientation}}},
      {{Direction<3>::upper_xi(), {4, quarter_turn_cw_about_zeta}},
       {Direction<3>::upper_eta(), {3, aligned_orientation}},
       {Direction<3>::lower_xi(), {5, quarter_turn_ccw_about_zeta}},
       {Direction<3>::lower_eta(), {2, aligned_orientation}}},
      {{Direction<3>::upper_xi(), {4, half_turn_about_zeta}},
       {Direction<3>::upper_eta(), {1, aligned_orientation}},
       {Direction<3>::lower_xi(), {5, half_turn_about_zeta}},
       {Direction<3>::lower_eta(), {0, aligned_orientation}}},
      {{Direction<3>::upper_xi(), {4, aligned_orientation}},
       {Direction<3>::upper_eta(), {0, aligned_orientation}},
       {Direction<3>::lower_xi(), {5, aligned_orientation}},
       {Direction<3>::lower_eta(), {1, aligned_orientation}}},
      {{Direction<3>::upper_xi(), {2, half_turn_about_zeta}},
       {Direction<3>::upper_eta(), {0, quarter_turn_cw_about_zeta}},
       {Direction<3>::lower_xi(), {3, aligned_orientation}},
       {Direction<3>::lower_eta(), {1, quarter_turn_ccw_about_zeta}}},
      {{Direction<3>::upper_xi(), {3, aligned_orientation}},
       {Direction<3>::upper_eta(), {0, quarter_turn_ccw_about_zeta}},
       {Direction<3>::lower_xi(), {2, half_turn_about_zeta}},
       {Direction<3>::lower_eta(), {1, quarter_turn_cw_about_zeta}}}};
  const std::vector<std::unordered_set<Direction<3>>>
      expected_external_boundaries{
          {{Direction<3>::upper_zeta()}, {Direction<3>::lower_zeta()}},
          {{Direction<3>::upper_zeta()}, {Direction<3>::lower_zeta()}},
          {{Direction<3>::upper_zeta()}, {Direction<3>::lower_zeta()}},
          {{Direction<3>::upper_zeta()}, {Direction<3>::lower_zeta()}},
          {{Direction<3>::upper_zeta()}, {Direction<3>::lower_zeta()}},
          {{Direction<3>::upper_zeta()}, {Direction<3>::lower_zeta()}}};

  const std::vector<std::array<size_t, 3>>& expected_extents{
      6,
      {{expected_shell_extents[1], expected_shell_extents[1],
        expected_shell_extents[0]}}};

  CHECK(shell.initial_extents() == expected_extents);
  CHECK(shell.initial_refinement_levels() == expected_refinement_level);
  using Wedge3DMap = CoordinateMaps::Wedge3D;

  test_domain_construction(
      domain, expected_block_neighbors, expected_external_boundaries,
      make_vector_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Wedge3DMap{inner_radius, outer_radius, OrientationMap<3>{}, 1.0, 1.0,
                     use_equiangular_map},
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                          Direction<3>::lower_zeta()}}},
                     1.0, 1.0, use_equiangular_map},
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
                          Direction<3>::lower_eta()}}},
                     1.0, 1.0, use_equiangular_map},
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
                          Direction<3>::upper_eta()}}},
                     1.0, 1.0, use_equiangular_map},
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::upper_zeta(), Direction<3>::upper_xi(),
                          Direction<3>::upper_eta()}}},
                     1.0, 1.0, use_equiangular_map},
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::lower_zeta(), Direction<3>::lower_xi(),
                          Direction<3>::upper_eta()}}},
                     1.0, 1.0, use_equiangular_map}

          ));

  test_initial_domain(domain, shell.initial_refinement_levels());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.Shell.Boundaries.Equiangular",
                  "[Domain][Unit]") {
  const double inner_radius = 1.0, outer_radius = 2.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points_r_angular{{4, 4}};

  const DomainCreators::Shell<Frame::Inertial> shell{
      inner_radius, outer_radius, refinement_level, grid_points_r_angular,
      true};
  test_physical_separation(shell.create_domain().blocks());
  test_shell_construction(shell, inner_radius, outer_radius, true,
                          grid_points_r_angular,
                          {6, make_array<3>(refinement_level)});
}

SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.Shell.Factory.Equiangular",
                  "[Domain][Unit]") {
  const auto shell = test_factory_creation<DomainCreator<3, Frame::Inertial>>(
      "  Shell:\n"
      "    InnerRadius: 1\n"
      "    OuterRadius: 3\n"
      "    InitialRefinement: 2\n"
      "    InitialGridPoints: [2,3]\n"
      "    UseEquiangularMap: true\n");
  const double inner_radius = 1.0, outer_radius = 3.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
  test_shell_construction(
      dynamic_cast<const DomainCreators::Shell<Frame::Inertial>&>(*shell),
      inner_radius, outer_radius, true, grid_points_r_angular,
      {6, make_array<3>(refinement_level)});
}

SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.Shell.Boundaries.Equidistant",
                  "[Domain][Unit]") {
  const double inner_radius = 1.0, outer_radius = 2.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points_r_angular{{4, 4}};

  const DomainCreators::Shell<Frame::Inertial> shell{
      inner_radius, outer_radius, refinement_level, grid_points_r_angular,
      false};
  test_physical_separation(shell.create_domain().blocks());
  test_shell_construction(shell, inner_radius, outer_radius, false,
                          grid_points_r_angular,
                          {6, make_array<3>(refinement_level)});
}

SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.Shell.Factory.Equidistant",
                  "[Domain][Unit]") {
  const auto shell = test_factory_creation<DomainCreator<3, Frame::Inertial>>(
      "  Shell:\n"
      "    InnerRadius: 1\n"
      "    OuterRadius: 3\n"
      "    InitialRefinement: 2\n"
      "    InitialGridPoints: [2,3]\n"
      "    UseEquiangularMap: false\n");
  const double inner_radius = 1.0, outer_radius = 3.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
  test_shell_construction(
      dynamic_cast<const DomainCreators::Shell<Frame::Inertial>&>(*shell),
      inner_radius, outer_radius, false, grid_points_r_angular,
      {6, make_array<3>(refinement_level)});
}
