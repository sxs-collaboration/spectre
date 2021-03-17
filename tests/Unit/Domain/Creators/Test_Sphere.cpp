// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/EquatorialCompression.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "Domain/OptionTags.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain {
namespace {
using BoundaryCondVector = std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>;

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_boundary_condition() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::upper_zeta(), 50);
}

std::string boundary_conditions_string() {
  return {
      "  BoundaryCondition:\n"
      "    TestBoundaryCondition:\n"
      "      Direction: upper-zeta\n"
      "      BlockId: 50\n"};
}

auto create_boundary_conditions() {
  BoundaryCondVector boundary_conditions_all_blocks{7};
  const auto boundary_condition = create_boundary_condition();
  for (size_t block_id = 0; block_id < 6; ++block_id) {
    boundary_conditions_all_blocks[block_id][Direction<3>::upper_zeta()] =
        boundary_condition->get_clone();
  }
  return boundary_conditions_all_blocks;
}

void test_sphere_construction(
    const creators::Sphere& sphere, const double inner_radius,
    const double outer_radius, const bool use_equiangular_map,
    const std::array<size_t, 2>& expected_sphere_extents,
    const std::vector<std::array<size_t, 3>>& expected_refinement_level,
    const BoundaryCondVector& expected_boundary_conditions = {}) {
  const auto domain = sphere.create_domain();
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
  const OrientationMap<3> center_relative_to_minus_z(
      std::array<Direction<3>, 3>{{Direction<3>::upper_xi(),
                                   Direction<3>::lower_eta(),
                                   Direction<3>::lower_zeta()}});
  const OrientationMap<3> center_relative_to_plus_y(std::array<Direction<3>, 3>{
      {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
       Direction<3>::upper_eta()}});
  const OrientationMap<3> center_relative_to_minus_y(
      std::array<Direction<3>, 3>{{Direction<3>::upper_xi(),
                                   Direction<3>::upper_zeta(),
                                   Direction<3>::lower_eta()}});
  const OrientationMap<3> center_relative_to_plus_x(std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::upper_zeta(),
       Direction<3>::upper_xi()}});
  const OrientationMap<3> center_relative_to_minus_x(
      std::array<Direction<3>, 3>{{Direction<3>::lower_eta(),
                                   Direction<3>::upper_zeta(),
                                   Direction<3>::lower_xi()}});

  const std::vector<DirectionMap<3, BlockNeighbor<3>>> expected_block_neighbors{
      {{Direction<3>::upper_xi(), {4, quarter_turn_ccw_about_zeta}},
       {Direction<3>::upper_eta(), {2, aligned_orientation}},
       {Direction<3>::lower_xi(), {5, quarter_turn_cw_about_zeta}},
       {Direction<3>::lower_eta(), {3, aligned_orientation}},
       {Direction<3>::lower_zeta(), {6, aligned_orientation}}},
      {{Direction<3>::upper_xi(), {4, quarter_turn_cw_about_zeta}},
       {Direction<3>::upper_eta(), {3, aligned_orientation}},
       {Direction<3>::lower_xi(), {5, quarter_turn_ccw_about_zeta}},
       {Direction<3>::lower_eta(), {2, aligned_orientation}},
       {Direction<3>::lower_zeta(), {6, center_relative_to_minus_z}}},
      {{Direction<3>::upper_xi(), {4, half_turn_about_zeta}},
       {Direction<3>::upper_eta(), {1, aligned_orientation}},
       {Direction<3>::lower_xi(), {5, half_turn_about_zeta}},
       {Direction<3>::lower_eta(), {0, aligned_orientation}},
       {Direction<3>::lower_zeta(), {6, center_relative_to_plus_y}}},
      {{Direction<3>::upper_xi(), {4, aligned_orientation}},
       {Direction<3>::upper_eta(), {0, aligned_orientation}},
       {Direction<3>::lower_xi(), {5, aligned_orientation}},
       {Direction<3>::lower_eta(), {1, aligned_orientation}},
       {Direction<3>::lower_zeta(), {6, center_relative_to_minus_y}}},
      {{Direction<3>::upper_xi(), {2, half_turn_about_zeta}},
       {Direction<3>::upper_eta(), {0, quarter_turn_cw_about_zeta}},
       {Direction<3>::lower_xi(), {3, aligned_orientation}},
       {Direction<3>::lower_eta(), {1, quarter_turn_ccw_about_zeta}},
       {Direction<3>::lower_zeta(), {6, center_relative_to_plus_x}}},
      {{Direction<3>::upper_xi(), {3, aligned_orientation}},
       {Direction<3>::upper_eta(), {0, quarter_turn_ccw_about_zeta}},
       {Direction<3>::lower_xi(), {2, half_turn_about_zeta}},
       {Direction<3>::lower_eta(), {1, quarter_turn_cw_about_zeta}},
       {Direction<3>::lower_zeta(), {6, center_relative_to_minus_x}}},
      {{Direction<3>::upper_zeta(), {0, aligned_orientation}},
       {Direction<3>::lower_zeta(),
        {1, center_relative_to_minus_z.inverse_map()}},
       {Direction<3>::upper_eta(),
        {2, center_relative_to_plus_y.inverse_map()}},
       {Direction<3>::lower_eta(),
        {3, center_relative_to_minus_y.inverse_map()}},
       {Direction<3>::upper_xi(), {4, center_relative_to_plus_x.inverse_map()}},
       {Direction<3>::lower_xi(),
        {5, center_relative_to_minus_x.inverse_map()}}}};

  const std::vector<std::unordered_set<Direction<3>>>
      expected_external_boundaries{{{Direction<3>::upper_zeta()}},
                                   {{Direction<3>::upper_zeta()}},
                                   {{Direction<3>::upper_zeta()}},
                                   {{Direction<3>::upper_zeta()}},
                                   {{Direction<3>::upper_zeta()}},
                                   {{Direction<3>::upper_zeta()}},
                                   {}};

  std::vector<std::array<size_t, 3>> expected_extents{
      6,
      {{expected_sphere_extents[1], expected_sphere_extents[1],
        expected_sphere_extents[0]}}};
  expected_extents.push_back(
      {{expected_sphere_extents[1], expected_sphere_extents[1],
        expected_sphere_extents[1]}});

  CHECK(sphere.initial_extents() == expected_extents);
  CHECK(sphere.initial_refinement_levels() == expected_refinement_level);
  using Wedge3DMap = CoordinateMaps::Wedge<3>;
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3D =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>;

  auto coord_maps =
      make_vector_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Wedge3DMap{inner_radius, outer_radius, OrientationMap<3>{}, 0.0, 1.0,
                     use_equiangular_map},
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                          Direction<3>::lower_zeta()}}},
                     0.0, 1.0, use_equiangular_map},
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
                          Direction<3>::lower_eta()}}},
                     0.0, 1.0, use_equiangular_map},
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
                          Direction<3>::upper_eta()}}},
                     0.0, 1.0, use_equiangular_map},
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::upper_zeta(), Direction<3>::upper_xi(),
                          Direction<3>::upper_eta()}}},
                     0.0, 1.0, use_equiangular_map},
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::lower_zeta(), Direction<3>::lower_xi(),
                          Direction<3>::upper_eta()}}},
                     0.0, 1.0, use_equiangular_map});
  if (use_equiangular_map) {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, Frame::Inertial>(Equiangular3D{
            Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(3.0),
                        inner_radius / sqrt(3.0)),
            Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(3.0),
                        inner_radius / sqrt(3.0)),
            Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(3.0),
                        inner_radius / sqrt(3.0))}));
  } else {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            Affine3D{Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(3.0),
                            inner_radius / sqrt(3.0)),
                     Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(3.0),
                            inner_radius / sqrt(3.0)),
                     Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(3.0),
                            inner_radius / sqrt(3.0))}));
  }
  test_domain_construction(domain, expected_block_neighbors,
                           expected_external_boundaries, coord_maps,
                           std::numeric_limits<double>::signaling_NaN(), {}, {},
                           expected_boundary_conditions);
  const auto coord_maps_copy = clone_unique_ptrs(coord_maps);

  Domain<3> domain_no_corners =
      expected_boundary_conditions.empty()
          ? Domain<3>{std::move(coord_maps)}
          : Domain<3>{std::move(coord_maps), create_boundary_conditions()};

  test_domain_construction(domain_no_corners, expected_block_neighbors,
                           expected_external_boundaries, coord_maps_copy);

  test_initial_domain(domain, sphere.initial_refinement_levels());
  test_initial_domain(domain_no_corners, sphere.initial_refinement_levels());

  Parallel::register_classes_in_list<
      typename domain::creators::Sphere::maps_list>();

  test_serialization(domain);
  test_serialization(domain_no_corners);
}

void test_sphere_boundaries_equiangular() {
  INFO("Sphere boundaries equiangular");
  const double inner_radius = 1.0;
  const double outer_radius = 2.0;
  const size_t refinement = 2;
  const std::array<size_t, 2> grid_points_r_angular{{4, 4}};

  const creators::Sphere sphere{inner_radius, outer_radius, refinement,
                                grid_points_r_angular, true};
  test_physical_separation(sphere.create_domain().blocks());

  test_sphere_construction(sphere, inner_radius, outer_radius, true,
                           grid_points_r_angular,
                           {7, make_array<3>(refinement)});

  const creators::Sphere sphere_boundary_condition{
      inner_radius,          outer_radius, refinement,
      grid_points_r_angular, true,         create_boundary_condition()};
  test_physical_separation(sphere_boundary_condition.create_domain().blocks());

  test_sphere_construction(sphere_boundary_condition, inner_radius,
                           outer_radius, true, grid_points_r_angular,
                           {7, make_array<3>(refinement)},
                           create_boundary_conditions());

  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, outer_radius, refinement, grid_points_r_angular, true,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Cannot have periodic boundary conditions with a Sphere"));
  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, outer_radius, refinement, grid_points_r_angular, true,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "None boundary condition is not supported. If you would like "
          "an outflow boundary condition, you must use that."));
}

void test_sphere_factory_equiangular() {
  INFO("Sphere factory equiangular");
  const auto helper = [](const auto expected_boundary_conditions,
                         auto use_boundary_condition) {
    const auto sphere = TestHelpers::test_factory_creation<
        DomainCreator<3>, domain::OptionTags::DomainCreator<3>,
        tmpl::conditional_t<decltype(use_boundary_condition)::value,
                            TestHelpers::domain::BoundaryConditions::
                                MetavariablesWithBoundaryConditions<3>,
                            TestHelpers::domain::BoundaryConditions::
                                MetavariablesWithoutBoundaryConditions<3>>>(
        "Sphere:\n"
        "  InnerRadius: 1\n"
        "  OuterRadius: 3\n"
        "  InitialRefinement: 2\n"
        "  InitialGridPoints: [2,3]\n"
        "  UseEquiangularMap: true\n" +
        (expected_boundary_conditions.empty() ? std::string{}
                                              : boundary_conditions_string()));
    const double inner_radius = 1.0;
    const double outer_radius = 3.0;
    const size_t refinement_level = 2;
    const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
    test_sphere_construction(
        dynamic_cast<const creators::Sphere&>(*sphere), inner_radius,
        outer_radius, true, grid_points_r_angular,
        {7, make_array<3>(refinement_level)}, expected_boundary_conditions);
  };
  helper(BoundaryCondVector{}, std::false_type{});
  helper(create_boundary_conditions(), std::true_type{});
}

void test_sphere_boundaries_equidistant() {
  INFO("Sphere boundaries equidistant");
  const double inner_radius = 1.0;
  const double outer_radius = 2.0;
  const size_t refinement = 2;
  const std::array<size_t, 2> grid_points_r_angular{{4, 4}};

  const creators::Sphere sphere{inner_radius, outer_radius, refinement,
                                grid_points_r_angular, false};
  test_physical_separation(sphere.create_domain().blocks());

  test_sphere_construction(sphere, inner_radius, outer_radius, false,
                           grid_points_r_angular,
                           {7, make_array<3>(refinement)});

  const creators::Sphere sphere_boundary_condition{
      inner_radius,          outer_radius, refinement,
      grid_points_r_angular, false,        create_boundary_condition()};
  test_physical_separation(sphere_boundary_condition.create_domain().blocks());

  test_sphere_construction(sphere_boundary_condition, inner_radius,
                           outer_radius, false, grid_points_r_angular,
                           {7, make_array<3>(refinement)},
                           create_boundary_conditions());
}

void test_sphere_factory_equidistant() {
  INFO("Sphere factory equidistant");
  const auto helper = [](const auto expected_boundary_conditions,
                         auto use_boundary_condition) {
    const auto sphere = TestHelpers::test_factory_creation<
        DomainCreator<3>, domain::OptionTags::DomainCreator<3>,
        tmpl::conditional_t<decltype(use_boundary_condition)::value,
                            TestHelpers::domain::BoundaryConditions::
                                MetavariablesWithBoundaryConditions<3>,
                            TestHelpers::domain::BoundaryConditions::
                                MetavariablesWithoutBoundaryConditions<3>>>(
        "Sphere:\n"
        "  InnerRadius: 1\n"
        "  OuterRadius: 3\n"
        "  InitialRefinement: 2\n"
        "  InitialGridPoints: [2,3]\n"
        "  UseEquiangularMap: false\n" +
        (expected_boundary_conditions.empty() ? std::string{}
                                              : boundary_conditions_string()));
    const double inner_radius = 1.0;
    const double outer_radius = 3.0;
    const size_t refinement_level = 2;
    const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
    test_sphere_construction(
        dynamic_cast<const creators::Sphere&>(*sphere), inner_radius,
        outer_radius, false, grid_points_r_angular,
        {7, make_array<3>(refinement_level)}, expected_boundary_conditions);
  };
  helper(BoundaryCondVector{}, std::false_type{});
  helper(create_boundary_conditions(), std::true_type{});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.Sphere", "[Domain][Unit]") {
  test_sphere_boundaries_equiangular();
  test_sphere_factory_equiangular();
  test_sphere_boundaries_equidistant();
  test_sphere_factory_equidistant();
}
}  // namespace domain
