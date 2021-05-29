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

#include "DataStructures/DataVector.hpp"
#include "DataStructures/IdPair.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/EquatorialCompression.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/OptionTags.hpp"
#include "Domain/Structure/BlockId.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare BlockNeighbor

namespace domain {
namespace {
using Translation = CoordinateMaps::TimeDependent::Translation;
using Translation3D =
    CoordinateMaps::TimeDependent::ProductOf3Maps<Translation, Translation,
                                                  Translation>;

using BoundaryCondVector = std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>;

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_inner_boundary_condition() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::lower_zeta(), 50);
}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_outer_boundary_condition() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::upper_zeta(), 50);
}

auto create_boundary_conditions(const size_t number_of_layers,
                                const ShellWedges wedges) {
  if (wedges != ShellWedges::All) {
    ERROR("Can only set boundary conditions when using all wedges, but got "
          << wedges);
  }
  BoundaryCondVector boundary_conditions_all_blocks{6 * number_of_layers};
  const auto inner_boundary_condition = create_inner_boundary_condition();
  const auto outer_boundary_condition = create_outer_boundary_condition();
  for (size_t block_id = 0; block_id < 6; ++block_id) {
    boundary_conditions_all_blocks[block_id][Direction<3>::lower_zeta()] =
        inner_boundary_condition->get_clone();
    boundary_conditions_all_blocks[6 * number_of_layers - block_id - 1]
                                  [Direction<3>::upper_zeta()] =
                                      outer_boundary_condition->get_clone();
  }
  return boundary_conditions_all_blocks;
}

template <typename... FuncsOfTime>
void test_shell_construction(
    const creators::Shell& shell, const double inner_radius,
    const double outer_radius, const bool use_equiangular_map,
    const std::array<size_t, 2>& expected_shell_extents,
    const std::vector<std::array<size_t, 3>>& expected_refinement_level,
    const double aspect_ratio = 1.0, const bool use_logarithmic_map = false,
    const ShellWedges which_wedges = ShellWedges::All,
    const std::tuple<std::pair<std::string, FuncsOfTime>...>&
        expected_functions_of_time = {},
    const std::vector<std::unique_ptr<domain::CoordinateMapBase<
        Frame::Grid, Frame::Inertial, 3>>>& expected_grid_to_inertial_maps = {},
    const std::vector<DirectionMap<
        3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>&
        expected_boundary_conditions = {}) {
  Parallel::register_classes_with_charm(
      typename domain::creators::Shell::maps_list{});
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
  std::vector<DirectionMap<3, BlockNeighbor<3>>> expected_block_neighbors{
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
  std::vector<std::unordered_set<Direction<3>>> expected_external_boundaries{
      {{Direction<3>::upper_zeta()}, {Direction<3>::lower_zeta()}},
      {{Direction<3>::upper_zeta()}, {Direction<3>::lower_zeta()}},
      {{Direction<3>::upper_zeta()}, {Direction<3>::lower_zeta()}},
      {{Direction<3>::upper_zeta()}, {Direction<3>::lower_zeta()}},
      {{Direction<3>::upper_zeta()}, {Direction<3>::lower_zeta()}},
      {{Direction<3>::upper_zeta()}, {Direction<3>::lower_zeta()}}};

  std::vector<std::array<size_t, 3>>::size_type num_pieces = 6;
  if (UNLIKELY(which_wedges == ShellWedges::FourOnEquator)) {
    num_pieces = 4;
    expected_block_neighbors = {
        {{Direction<3>::upper_xi(), {2, half_turn_about_zeta}},
         {Direction<3>::lower_xi(), {3, half_turn_about_zeta}}},
        {{Direction<3>::upper_xi(), {2, aligned_orientation}},
         {Direction<3>::lower_xi(), {3, aligned_orientation}}},
        {{Direction<3>::upper_xi(), {0, half_turn_about_zeta}},
         {Direction<3>::lower_xi(), {1, aligned_orientation}}},
        {{Direction<3>::upper_xi(), {1, aligned_orientation}},
         {Direction<3>::lower_xi(), {0, half_turn_about_zeta}}}};
    expected_external_boundaries = {{{Direction<3>::upper_zeta()},
                                     {Direction<3>::lower_zeta()},
                                     {Direction<3>::upper_eta()},
                                     {Direction<3>::lower_eta()}},
                                    {{Direction<3>::upper_zeta()},
                                     {Direction<3>::lower_zeta()},
                                     {Direction<3>::upper_eta()},
                                     {Direction<3>::lower_eta()}},
                                    {{Direction<3>::upper_zeta()},
                                     {Direction<3>::lower_zeta()},
                                     {Direction<3>::upper_eta()},
                                     {Direction<3>::lower_eta()}},
                                    {{Direction<3>::upper_zeta()},
                                     {Direction<3>::lower_zeta()},
                                     {Direction<3>::upper_eta()},
                                     {Direction<3>::lower_eta()}}};

  } else if (UNLIKELY(which_wedges == ShellWedges::OneAlongMinusX)) {
    num_pieces = 1;
    expected_block_neighbors = {{}};
    expected_external_boundaries = {{{Direction<3>::upper_zeta()},
                                     {Direction<3>::lower_zeta()},
                                     {Direction<3>::upper_eta()},
                                     {Direction<3>::lower_eta()},
                                     {Direction<3>::upper_xi()},
                                     {Direction<3>::lower_xi()}}};
  }
  const std::vector<std::array<size_t, 3>>& expected_extents{
      num_pieces,
      {{expected_shell_extents[1], expected_shell_extents[1],
        expected_shell_extents[0]}}};

  CHECK(shell.initial_extents() == expected_extents);
  CHECK(shell.initial_refinement_levels() == expected_refinement_level);
  using Wedge3DMap = CoordinateMaps::Wedge<3>;
  using Halves = Wedge3DMap::WedgeHalves;
  const auto radial_distribution =
      use_logarithmic_map ? domain::CoordinateMaps::Distribution::Logarithmic
                          : domain::CoordinateMaps::Distribution::Linear;
  if (aspect_ratio == 1.0) {
    auto vector_of_maps = make_vector_coordinate_map_base<
        Frame::Logical, tmpl::conditional_t<sizeof...(FuncsOfTime) == 0,
                                            Frame::Inertial, Frame::Grid>>(
        Wedge3DMap{inner_radius, outer_radius, 1.0, 1.0, OrientationMap<3>{},
                   use_equiangular_map, Halves::Both, radial_distribution},
        Wedge3DMap{inner_radius, outer_radius, 1.0, 1.0,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                        Direction<3>::lower_zeta()}}},
                   use_equiangular_map, Halves::Both, radial_distribution},
        Wedge3DMap{inner_radius, outer_radius, 1.0, 1.0,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
                        Direction<3>::lower_eta()}}},
                   use_equiangular_map, Halves::Both, radial_distribution},
        Wedge3DMap{inner_radius, outer_radius, 1.0, 1.0,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
                        Direction<3>::upper_eta()}}},
                   use_equiangular_map, Halves::Both, radial_distribution},
        Wedge3DMap{inner_radius, outer_radius, 1.0, 1.0,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_zeta(), Direction<3>::upper_xi(),
                        Direction<3>::upper_eta()}}},
                   use_equiangular_map, Halves::Both, radial_distribution},
        Wedge3DMap{inner_radius, outer_radius, 1.0, 1.0,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::lower_zeta(), Direction<3>::lower_xi(),
                        Direction<3>::upper_eta()}}},
                   use_equiangular_map, Halves::Both, radial_distribution});
    if (UNLIKELY(which_wedges == ShellWedges::FourOnEquator)) {
      vector_of_maps.erase(vector_of_maps.begin(), vector_of_maps.begin() + 2);
    } else if (UNLIKELY(which_wedges == ShellWedges::OneAlongMinusX)) {
      vector_of_maps.erase(vector_of_maps.begin(), vector_of_maps.begin() + 5);
    }
    test_domain_construction(
        domain, expected_block_neighbors, expected_external_boundaries,
        vector_of_maps,
        expected_grid_to_inertial_maps.empty() ?
        std::numeric_limits<double>::signaling_NaN() : 10.0,
        shell.functions_of_time(), expected_grid_to_inertial_maps,
        expected_boundary_conditions);

    if constexpr(sizeof...(FuncsOfTime) == 0) {
      // We turn off the domain_no_corners test for time-dependent
      // maps because Domain doesn't have a constructor that takes
      // maps from Logical to Grid.
      const auto vector_of_maps_copy = clone_unique_ptrs(vector_of_maps);

      Domain<3> domain_no_corners =
          expected_boundary_conditions.empty()
              ? Domain<3>{std::move(vector_of_maps)}
              : Domain<3>{std::move(vector_of_maps),
                          create_boundary_conditions(
                              expected_boundary_conditions.size() / 6,
                              ShellWedges::All)};

      test_domain_construction(
          domain_no_corners, expected_block_neighbors,
          expected_external_boundaries, vector_of_maps_copy,
          expected_grid_to_inertial_maps.empty()
              ? std::numeric_limits<double>::signaling_NaN()
              : 10.0,
          shell.functions_of_time(), expected_grid_to_inertial_maps,
          expected_boundary_conditions);
      test_initial_domain(domain_no_corners, shell.initial_refinement_levels());
      test_serialization(domain_no_corners);
    }
  } else {
    const auto compression =
        CoordinateMaps::EquatorialCompression{aspect_ratio};
    // Set up translation map:
    using Identity2D = domain::CoordinateMaps::Identity<2>;
    using Affine = domain::CoordinateMaps::Affine;
    const auto translation =
        domain::CoordinateMaps::ProductOf2Maps<Affine, Identity2D>(
            Affine{-1.0, 1.0, -1.0, 1.0}, Identity2D{});
    using TargetFrame = tmpl::conditional_t<sizeof...(FuncsOfTime) == 0,
                                            Frame::Inertial, Frame::Grid>;
    auto vector_of_maps = make_vector(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Wedge3DMap{inner_radius, outer_radius, 1.0, 1.0,
                       OrientationMap<3>{}, use_equiangular_map, Halves::Both,
                       radial_distribution},
            compression, translation),
        make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Wedge3DMap{inner_radius, outer_radius, 1.0, 1.0,
                       OrientationMap<3>{std::array<Direction<3>, 3>{
                           {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                            Direction<3>::lower_zeta()}}},
                       use_equiangular_map, Halves::Both, radial_distribution},
            compression, translation),
        make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Wedge3DMap{
                inner_radius, outer_radius, 1.0, 1.0,
                OrientationMap<3>{std::array<Direction<3>, 3>{
                    {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
                     Direction<3>::lower_eta()}}},
                use_equiangular_map, Halves::Both, radial_distribution},
            compression, translation),
        make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Wedge3DMap{
                inner_radius, outer_radius, 1.0, 1.0,
                OrientationMap<3>{std::array<Direction<3>, 3>{
                    {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
                     Direction<3>::upper_eta()}}},
                use_equiangular_map, Halves::Both, radial_distribution},
            compression, translation),
        make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Wedge3DMap{
                inner_radius, outer_radius, 1.0, 1.0,
                OrientationMap<3>{std::array<Direction<3>, 3>{
                    {Direction<3>::upper_zeta(), Direction<3>::upper_xi(),
                     Direction<3>::upper_eta()}}},
                use_equiangular_map, Halves::Both, radial_distribution},
            compression, translation),
        make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Wedge3DMap{
                inner_radius, outer_radius, 1.0, 1.0,
                OrientationMap<3>{std::array<Direction<3>, 3>{
                    {Direction<3>::lower_zeta(), Direction<3>::lower_xi(),
                     Direction<3>::upper_eta()}}},
                use_equiangular_map, Halves::Both, radial_distribution},
            compression, translation));
    if (UNLIKELY(which_wedges == ShellWedges::FourOnEquator)) {
      vector_of_maps.erase(vector_of_maps.begin(), vector_of_maps.begin() + 2);
    } else if (UNLIKELY(which_wedges == ShellWedges::OneAlongMinusX)) {
      vector_of_maps.erase(vector_of_maps.begin(), vector_of_maps.begin() + 5);
    }
    test_domain_construction(domain, expected_block_neighbors,
                             expected_external_boundaries, vector_of_maps,
                             expected_grid_to_inertial_maps.empty()
                                 ? std::numeric_limits<double>::signaling_NaN()
                                 : 10.0,
                             shell.functions_of_time(),
                             expected_grid_to_inertial_maps,
                             expected_boundary_conditions);

    if constexpr (sizeof...(FuncsOfTime) == 0) {
      // We turn off the domain_no_corners test for time-dependent
      // maps because Domain doesn't have a constructor that takes
      // maps from Logical to Grid.
      const auto vector_of_maps_copy = clone_unique_ptrs(vector_of_maps);
      Domain<3> domain_no_corners =
          expected_boundary_conditions.empty()
              ? Domain<3>{std::move(vector_of_maps)}
              : Domain<3>{std::move(vector_of_maps),
                          create_boundary_conditions(
                              expected_boundary_conditions.size() / 6,
                              ShellWedges::All)};
      test_domain_construction(
          domain_no_corners, expected_block_neighbors,
          expected_external_boundaries, vector_of_maps_copy,
          expected_grid_to_inertial_maps.empty() == 0
              ? std::numeric_limits<double>::signaling_NaN()
              : 10.0,
          shell.functions_of_time(), expected_grid_to_inertial_maps,
          expected_boundary_conditions);

      test_initial_domain(domain_no_corners, shell.initial_refinement_levels());
      test_serialization(domain_no_corners);
    }
  }

  test_initial_domain(domain, shell.initial_refinement_levels());
  TestHelpers::domain::creators::test_functions_of_time(
      shell, expected_functions_of_time);
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  test_serialization(domain);
}

void test_shell_boundaries() {
  INFO("Shell boundaries");
  const double inner_radius = 1.0;
  const double outer_radius = 2.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points_r_angular{{4, 4}};

  for (const auto& use_equiangular_map : {true, false}) {
    const creators::Shell shell{inner_radius, outer_radius, refinement_level,
                                grid_points_r_angular, use_equiangular_map};
    test_physical_separation(shell.create_domain().blocks());
    test_shell_construction(shell, inner_radius, outer_radius,
                            use_equiangular_map, grid_points_r_angular,
                            {6, make_array<3>(refinement_level)});

    const creators::Shell shell_boundary_conditions{
        inner_radius,
        outer_radius,
        refinement_level,
        grid_points_r_angular,
        use_equiangular_map,
        1.0,
        {},
        {domain::CoordinateMaps::Distribution::Linear},
        ShellWedges::All,
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>{},
        create_inner_boundary_condition(),
        create_outer_boundary_condition()};
    test_physical_separation(
        shell_boundary_conditions.create_domain().blocks());
    test_shell_construction(
        shell_boundary_conditions, inner_radius, outer_radius,
        use_equiangular_map, grid_points_r_angular,
        {6, make_array<3>(refinement_level)}, 1.0, false, ShellWedges::All, {},
        {}, create_boundary_conditions(1, ShellWedges::All));
  }
}

std::string boundary_conditions_string() {
  return {
      "  BoundaryConditions:\n"
      "    InnerBoundary:\n"
      "      TestBoundaryCondition:\n"
      "        Direction: lower-zeta\n"
      "        BlockId: 50\n"
      "    OuterBoundary:\n"
      "      TestBoundaryCondition:\n"
      "        Direction: upper-zeta\n"
      "        BlockId: 50\n"};
}

void test_shell_factory_equiangular() {
  INFO("Shell factory equiangular");
  const auto helper = [](const auto expected_boundary_conditions,
                         auto use_boundary_condition) {
    const auto shell = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<3>,
        tmpl::conditional_t<decltype(use_boundary_condition)::value,
                            TestHelpers::domain::BoundaryConditions::
                                MetavariablesWithBoundaryConditions<3>,
                            TestHelpers::domain::BoundaryConditions::
                                MetavariablesWithoutBoundaryConditions<3>>>(
        "Shell:\n"
        "  InnerRadius: 1\n"
        "  OuterRadius: 3\n"
        "  InitialRefinement: 2\n"
        "  InitialGridPoints: [2,3]\n"
        "  UseEquiangularMap: true\n"
        "  AspectRatio: 1.0\n"
        "  RadialPartitioning: []\n"
        "  RadialDistribution: [Linear]\n"
        "  WhichWedges: All\n"
        "  TimeDependence: None\n" +
        (expected_boundary_conditions.empty() ? std::string{}
                                              : boundary_conditions_string()));
    const double inner_radius = 1.0;
    const double outer_radius = 3.0;
    const size_t refinement_level = 2;
    const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
    test_shell_construction(
        dynamic_cast<const creators::Shell&>(*shell), inner_radius,
        outer_radius, true, grid_points_r_angular,
        {6, make_array<3>(refinement_level)}, 1.0, false, ShellWedges::All, {},
        {}, expected_boundary_conditions);
  };
  helper(BoundaryCondVector{}, std::false_type{});
  helper(create_boundary_conditions(1, ShellWedges::All), std::true_type{});
}

void test_shell_factory_equiangular_time_dependent() {
  INFO("Shell factory equiangular_time_dependent");
  const auto helper = [](const auto expected_boundary_conditions,
                         auto use_boundary_condition) {
    const auto shell = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<3>,
        tmpl::conditional_t<decltype(use_boundary_condition)::value,
                            TestHelpers::domain::BoundaryConditions::
                                MetavariablesWithBoundaryConditions<3>,
                            TestHelpers::domain::BoundaryConditions::
                                MetavariablesWithoutBoundaryConditions<3>>>(
        "Shell:\n"
        "  InnerRadius: 1\n"
        "  OuterRadius: 3\n"
        "  InitialRefinement: 2\n"
        "  InitialGridPoints: [2,3]\n"
        "  UseEquiangularMap: true\n"
        "  AspectRatio: 1.0\n"
        "  RadialPartitioning: []\n"
        "  RadialDistribution: [Linear]\n"
        "  WhichWedges: All\n"
        "  TimeDependence:\n"
        "    UniformTranslation:\n"
        "      InitialTime: 1.0\n"
        "      InitialExpirationDeltaT: 9.0\n"
        "      Velocity: [2.3, -0.3, 1.2]\n"
        "      FunctionOfTimeNames: [TranslationX,"
        "                            TranslationY,"
        "                            TranslationZ]\n" +
        (expected_boundary_conditions.empty() ? std::string{}
                                              : boundary_conditions_string()));
    const double inner_radius = 1.0;
    const double outer_radius = 3.0;
    const size_t refinement_level = 2;
    const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
    test_shell_construction(
        dynamic_cast<const creators::Shell&>(*shell), inner_radius,
        outer_radius, true, grid_points_r_angular,
        {6, make_array<3>(refinement_level)}, 1.0, false, ShellWedges::All,
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                "TranslationX",
                {1.0, std::array<DataVector, 3>{{{0.0}, {2.3}, {0.0}}}, 10.0}},
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                "TranslationY",
                {1.0, std::array<DataVector, 3>{{{0.0}, {-0.3}, {0.0}}}, 10.0}},
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                "TranslationZ",
                {1.0, std::array<DataVector, 3>{{{0.0}, {1.2}, {0.0}}}, 10.0}}),
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            Translation3D{Translation{"TranslationX"},
                          Translation{"TranslationY"},
                          Translation{"TranslationZ"}},
            Translation3D{Translation{"TranslationX"},
                          Translation{"TranslationY"},
                          Translation{"TranslationZ"}},
            Translation3D{Translation{"TranslationX"},
                          Translation{"TranslationY"},
                          Translation{"TranslationZ"}},
            Translation3D{Translation{"TranslationX"},
                          Translation{"TranslationY"},
                          Translation{"TranslationZ"}},
            Translation3D{Translation{"TranslationX"},
                          Translation{"TranslationY"},
                          Translation{"TranslationZ"}},
            Translation3D{Translation{"TranslationX"},
                          Translation{"TranslationY"},
                          Translation{"TranslationZ"}}),
        expected_boundary_conditions);
  };
  helper(BoundaryCondVector{}, std::false_type{});
  helper(create_boundary_conditions(1, ShellWedges::All), std::true_type{});
}

void test_shell_factory_equidistant() {
  INFO("Shell factory equidistant");
  const auto helper = [](const auto expected_boundary_conditions,
                         auto use_boundary_condition) {
    const auto shell = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<3>,
        tmpl::conditional_t<decltype(use_boundary_condition)::value,
                            TestHelpers::domain::BoundaryConditions::
                                MetavariablesWithBoundaryConditions<3>,
                            TestHelpers::domain::BoundaryConditions::
                                MetavariablesWithoutBoundaryConditions<3>>>(
        "Shell:\n"
        "  InnerRadius: 1\n"
        "  OuterRadius: 3\n"
        "  InitialRefinement: 2\n"
        "  InitialGridPoints: [2,3]\n"
        "  UseEquiangularMap: false\n"
        "  AspectRatio: 1.0\n"
        "  RadialPartitioning: []\n"
        "  RadialDistribution: [Linear]\n"
        "  WhichWedges: All\n"
        "  TimeDependence: None\n" +
        (expected_boundary_conditions.empty() ? std::string{}
                                              : boundary_conditions_string()));
    const double inner_radius = 1.0;
    const double outer_radius = 3.0;
    const size_t refinement_level = 2;
    const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
    test_shell_construction(
        dynamic_cast<const creators::Shell&>(*shell), inner_radius,
        outer_radius, false, grid_points_r_angular,
        {6, make_array<3>(refinement_level)}, 1.0, false, ShellWedges::All, {},
        {}, expected_boundary_conditions);
  };
  helper(BoundaryCondVector{}, std::false_type{});
  helper(create_boundary_conditions(1, ShellWedges::All), std::true_type{});
}

void test_shell_boundaries_aspect_ratio() {
  INFO("Shell boundaries aspect ratio");
  const double inner_radius = 1.0;
  const double outer_radius = 2.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points_r_angular{{4, 4}};
  const double aspect_ratio = 1.3;

  const creators::Shell shell{
      inner_radius,          outer_radius, refinement_level,
      grid_points_r_angular, false,        aspect_ratio};
  test_physical_separation(shell.create_domain().blocks());
  test_shell_construction(shell, inner_radius, outer_radius, false,
                          grid_points_r_angular,
                          {6, make_array<3>(refinement_level)}, aspect_ratio);

  const creators::Shell shell_boundary_condition{
      inner_radius,
      outer_radius,
      refinement_level,
      grid_points_r_angular,
      false,
      aspect_ratio,
      {},
      {domain::CoordinateMaps::Distribution::Linear},
      ShellWedges::All,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>{},
      create_inner_boundary_condition(),
      create_outer_boundary_condition()};
  test_physical_separation(shell_boundary_condition.create_domain().blocks());
  test_shell_construction(shell_boundary_condition, inner_radius, outer_radius,
                          false, grid_points_r_angular,
                          {6, make_array<3>(refinement_level)}, aspect_ratio,
                          false, ShellWedges::All, {}, {},
                          create_boundary_conditions(1, ShellWedges::All));
}

void test_shell_factory_aspect_ratio() {
  INFO("Shell factory aspect ratio");
  const auto helper = [](const auto expected_boundary_conditions,
                         auto use_boundary_condition) {
    const auto shell = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<3>,
        tmpl::conditional_t<decltype(use_boundary_condition)::value,
                            TestHelpers::domain::BoundaryConditions::
                                MetavariablesWithBoundaryConditions<3>,
                            TestHelpers::domain::BoundaryConditions::
                                MetavariablesWithoutBoundaryConditions<3>>>(
        "Shell:\n"
        "  InnerRadius: 1\n"
        "  OuterRadius: 3\n"
        "  InitialRefinement: 2\n"
        "  InitialGridPoints: [2,3]\n"
        "  UseEquiangularMap: false\n"
        "  AspectRatio: 2.0        \n"
        "  RadialPartitioning: []\n"
        "  RadialDistribution: [Linear]\n"
        "  WhichWedges: All\n"
        "  TimeDependence: None\n" +
        (expected_boundary_conditions.empty() ? std::string{}
                                              : boundary_conditions_string()));
    const double inner_radius = 1.0;
    const double outer_radius = 3.0;
    const size_t refinement_level = 2;
    const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
    const double aspect_ratio = 2.0;
    test_shell_construction(
        dynamic_cast<const creators::Shell&>(*shell), inner_radius,
        outer_radius, false, grid_points_r_angular,
        {6, make_array<3>(refinement_level)}, aspect_ratio, false,
        ShellWedges::All, {}, {}, expected_boundary_conditions);
  };
  helper(BoundaryCondVector{}, std::false_type{});
  helper(create_boundary_conditions(1, ShellWedges::All), std::true_type{});
}

void test_shell_boundaries_logarithmic_map() {
  INFO("Shell boundaries logarithmic map");
  const double inner_radius = 1.0;
  const double outer_radius = 2.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points_r_angular{{4, 4}};
  const double aspect_ratio = 1.0;
  const bool use_logarithmic_map = true;
  const std::vector<domain::CoordinateMaps::Distribution> radial_distribution{
      domain::CoordinateMaps::Distribution::Logarithmic};

  const creators::Shell shell{inner_radius,
                              outer_radius,
                              refinement_level,
                              grid_points_r_angular,
                              false,
                              aspect_ratio,
                              {},
                              radial_distribution};
  test_physical_separation(shell.create_domain().blocks());
  test_shell_construction(
      shell, inner_radius, outer_radius, false, grid_points_r_angular,
      {6, make_array<3>(refinement_level)}, aspect_ratio, use_logarithmic_map);

  const creators::Shell shell_boundary_condition{
      inner_radius,
      outer_radius,
      refinement_level,
      grid_points_r_angular,
      false,
      aspect_ratio,
      {},
      radial_distribution,
      ShellWedges::All,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>{},
      create_inner_boundary_condition(),
      create_outer_boundary_condition()};
  test_physical_separation(shell_boundary_condition.create_domain().blocks());
  test_shell_construction(shell_boundary_condition, inner_radius, outer_radius,
                          false, grid_points_r_angular,
                          {6, make_array<3>(refinement_level)}, aspect_ratio,
                          use_logarithmic_map, ShellWedges::All, {}, {},
                          create_boundary_conditions(1, ShellWedges::All));

  CHECK_THROWS_WITH(
      creators::Shell(
          inner_radius, outer_radius, refinement_level, grid_points_r_angular,
          false, aspect_ratio, {}, radial_distribution, ShellWedges::All,
          std::unique_ptr<
              domain::creators::time_dependence::TimeDependence<3>>{},
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Cannot have periodic boundary conditions with a shell"));
  CHECK_THROWS_WITH(
      creators::Shell(
          inner_radius, outer_radius, refinement_level, grid_points_r_angular,
          false, aspect_ratio, {}, radial_distribution, ShellWedges::All,
          std::unique_ptr<
              domain::creators::time_dependence::TimeDependence<3>>{},
          create_inner_boundary_condition(),
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Cannot have periodic boundary conditions with a shell"));
  CHECK_THROWS_WITH(
      creators::Shell(
          inner_radius, outer_radius, refinement_level, grid_points_r_angular,
          false, aspect_ratio, {}, radial_distribution, ShellWedges::All,
          std::unique_ptr<
              domain::creators::time_dependence::TimeDependence<3>>{},
          create_inner_boundary_condition(),
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "None boundary condition is not supported. If you would like "
          "an outflow boundary condition, you must use that."));
  CHECK_THROWS_WITH(
      creators::Shell(
          inner_radius, outer_radius, refinement_level, grid_points_r_angular,
          false, aspect_ratio, {}, radial_distribution, ShellWedges::All,
          std::unique_ptr<
              domain::creators::time_dependence::TimeDependence<3>>{},
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "None boundary condition is not supported. If you would like "
          "an outflow boundary condition, you must use that."));
}

void test_shell_factory_logarithmic_map() {
  INFO("Shell factory logarithmic map");
  const auto helper = [](const auto expected_boundary_conditions,
                         auto use_boundary_condition) {
    const auto shell = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<3>,
        tmpl::conditional_t<decltype(use_boundary_condition)::value,
                            TestHelpers::domain::BoundaryConditions::
                                MetavariablesWithBoundaryConditions<3>,
                            TestHelpers::domain::BoundaryConditions::
                                MetavariablesWithoutBoundaryConditions<3>>>(
        "Shell:\n"
        "  InnerRadius: 1\n"
        "  OuterRadius: 3\n"
        "  InitialRefinement: 2\n"
        "  InitialGridPoints: [2,3]\n"
        "  UseEquiangularMap: false\n"
        "  AspectRatio: 2.0        \n"
        "  RadialPartitioning: []\n"
        "  RadialDistribution: [Logarithmic]\n"
        "  WhichWedges: All\n"
        "  TimeDependence: None\n" +
        (expected_boundary_conditions.empty() ? std::string{}
                                              : boundary_conditions_string()));
    const double inner_radius = 1.0;
    const double outer_radius = 3.0;
    const size_t refinement_level = 2;
    const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
    const double aspect_ratio = 2.0;
    const bool use_logarithmic_map = true;
    test_shell_construction(
        dynamic_cast<const creators::Shell&>(*shell), inner_radius,
        outer_radius, false, grid_points_r_angular,
        {6, make_array<3>(refinement_level)}, aspect_ratio, use_logarithmic_map,
        ShellWedges::All, {}, {}, expected_boundary_conditions);
  };
  helper(BoundaryCondVector{}, std::false_type{});
  helper(create_boundary_conditions(1, ShellWedges::All), std::true_type{});

  INFO("Test with multiple radial layers");
  const auto shell =
      TestHelpers::test_option_tag<domain::OptionTags::DomainCreator<3>,
                                   TestHelpers::domain::BoundaryConditions::
                                       MetavariablesWithBoundaryConditions<3>>(
          "Shell:\n"
          "  InnerRadius: 1\n"
          "  OuterRadius: 3\n"
          "  InitialRefinement: 2\n"
          "  InitialGridPoints: [2,3]\n"
          "  UseEquiangularMap: false\n"
          "  AspectRatio: 2.0        \n"
          "  RadialPartitioning: [1.5, 2.5]\n"
          "  RadialDistribution: [Logarithmic, Logarithmic, Logarithmic]\n"
          "  WhichWedges: All\n"
          "  TimeDependence: None\n" +
          boundary_conditions_string());
  const Domain<3> multiple_layers_domain = shell->create_domain();
  const auto expected_boundary_conditions =
      create_boundary_conditions(3, ShellWedges::All);
  const auto& blocks = multiple_layers_domain.blocks();
  REQUIRE(expected_boundary_conditions.size() == blocks.size());
  for (size_t block_id = 0; block_id < expected_boundary_conditions.size();
       ++block_id) {
    CAPTURE(block_id);
    REQUIRE(blocks[block_id].external_boundary_conditions().size() ==
            expected_boundary_conditions[block_id].size());
    REQUIRE(blocks[block_id].external_boundary_conditions().size() ==
            blocks[block_id].external_boundaries().size());
    for (const Direction<3>& direction :
         blocks[block_id].external_boundaries()) {
      CAPTURE(direction);
      REQUIRE(blocks[block_id].external_boundary_conditions().count(
                  direction) == 1);
      REQUIRE(expected_boundary_conditions[block_id].count(direction) == 1);
      using BcType =
          TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>;
      const auto& expected_bc = dynamic_cast<const BcType&>(
          *expected_boundary_conditions[block_id].at(direction));
      const auto& bc = dynamic_cast<const BcType&>(
          *blocks[block_id].external_boundary_conditions().at(direction));
      CHECK(expected_bc.direction() == bc.direction());
      CHECK(expected_bc.block_id() == bc.block_id());
    }
  }
}

void test_shell_factory_wedges_four_on_equator() {
  INFO("Shell factory wedges four on equator");
  const auto shell = TestHelpers::test_option_tag<
      domain::OptionTags::DomainCreator<3>,
      TestHelpers::domain::BoundaryConditions::
          MetavariablesWithoutBoundaryConditions<3>>(
      "Shell:\n"
      "  InnerRadius: 1\n"
      "  OuterRadius: 3\n"
      "  InitialRefinement: 2\n"
      "  InitialGridPoints: [2,3]\n"
      "  UseEquiangularMap: false\n"
      "  AspectRatio: 2.0        \n"
      "  RadialPartitioning: []\n"
      "  RadialDistribution: [Logarithmic]\n"
      "  WhichWedges: FourOnEquator\n"
      "  TimeDependence: None\n");
  const double inner_radius = 1.0;
  const double outer_radius = 3.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
  const double aspect_ratio = 2.0;
  const bool use_logarithmic_map = true;
  const ShellWedges which_wedges = ShellWedges::FourOnEquator;
  test_shell_construction(
      dynamic_cast<const creators::Shell&>(*shell), inner_radius, outer_radius,
      false, grid_points_r_angular, {4, make_array<3>(refinement_level)},
      aspect_ratio, use_logarithmic_map, which_wedges);
}

void test_shell_factory_wedges_one_along_minus_x() {
  INFO("Shell factory wedges one along minus x");
  const auto shell = TestHelpers::test_option_tag<
      domain::OptionTags::DomainCreator<3>,
      TestHelpers::domain::BoundaryConditions::
          MetavariablesWithoutBoundaryConditions<3>>(
      "Shell:\n"
      "  InnerRadius: 2\n"
      "  OuterRadius: 3\n"
      "  InitialRefinement: 2\n"
      "  InitialGridPoints: [2,3]\n"
      "  UseEquiangularMap: true\n"
      "  AspectRatio: 2.7        \n"
      "  RadialPartitioning: []\n"
      "  RadialDistribution: [Linear]\n"
      "  WhichWedges: OneAlongMinusX \n"
      "  TimeDependence: None\n");
  const double inner_radius = 2.0;
  const double outer_radius = 3.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
  const double aspect_ratio = 2.7;
  const bool use_logarithmic_map = false;
  const ShellWedges which_wedges = ShellWedges::OneAlongMinusX;
  test_shell_construction(
      dynamic_cast<const creators::Shell&>(*shell), inner_radius, outer_radius,
      true, grid_points_r_angular, {1, make_array<3>(refinement_level)},
      aspect_ratio, use_logarithmic_map, which_wedges);
}

// Tests that the underlying element structure is the same regardless of
// whether they are created through InitialRefinement or RadialBlockLayers.
// This is done by constructing a set of expected boundary points dependent on
// only the total number of Elements in the Domain. The Domain is then created
// using either InitialRefinement or RadialBlockLayers. Comparing the resulting
// Element boundary points against the expected points ensures that both methods
// of creating Elements lead to the same boundary locations.
// To verify that the Block structures created are indeed distinct, this test
// also checks the Block structure.
void test_radial_block_layers(const double inner_radius,
                              const double outer_radius,
                              const size_t refinement_level,
                              const bool use_logarithmic_map,
                              const size_t radial_block_layers,
                              const std::vector<size_t>& expected_block_ids) {
  const size_t number_of_divisions =
      radial_block_layers * two_to_the(refinement_level);
  const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
  const bool use_equiangular_map = true;
  const double aspect_ratio = 1.0;
  const ShellWedges which_wedges = ShellWedges::OneAlongMinusX;
  // Points on the interior of Blocks used to check Block structure:
  DataVector x_in_block_interior(number_of_divisions);
  // Points on the boundary of Elements used to check Element structure:
  DataVector x_on_element_boundary(number_of_divisions + 1);
  for (size_t i = 0; i < number_of_divisions; i++) {
    const double delta_boundary =
        static_cast<double>(i) / static_cast<double>(number_of_divisions);
    const double delta_interior = (static_cast<double>(i) + 0.5) /
                                  static_cast<double>(number_of_divisions);
    if (use_logarithmic_map) {
      x_in_block_interior[i] =
          inner_radius * pow(outer_radius / inner_radius, delta_interior);
      x_on_element_boundary[i] =
          inner_radius * pow(outer_radius / inner_radius, delta_boundary);
    } else {
      x_in_block_interior[i] =
          inner_radius + delta_interior * (outer_radius - inner_radius);
      x_on_element_boundary[i] =
          inner_radius + delta_boundary * (outer_radius - inner_radius);
    }
  }
  x_on_element_boundary[number_of_divisions] = outer_radius;

  std::vector<double> radial_partitions(radial_block_layers - 1);
  for (size_t i = 1; i < radial_block_layers; ++i) {
    const double delta =
        static_cast<double>(i) / static_cast<double>(radial_block_layers);
    if (use_logarithmic_map) {
      radial_partitions[i - 1] =
          inner_radius * pow(outer_radius / inner_radius, delta);
    } else {
      radial_partitions[i - 1] =
          inner_radius + delta * (outer_radius - inner_radius);
    }
  }

  const auto zero = make_with_value<DataVector>(x_in_block_interior, 0.0);
  tnsr::I<DataVector, 3, Frame::Inertial> interior_inertial_coords{
      {{-x_in_block_interior, zero, zero}}};
  const std::vector<domain::CoordinateMaps::Distribution> radial_distribution(
      radial_block_layers,
      use_logarithmic_map ? domain::CoordinateMaps::Distribution::Logarithmic
                          : domain::CoordinateMaps::Distribution::Linear);
  const creators::Shell shell{
      inner_radius,          outer_radius,        refinement_level,
      grid_points_r_angular, use_equiangular_map, aspect_ratio,
      radial_partitions,     radial_distribution, which_wedges};
  auto domain = shell.create_domain();
  const auto blogical_coords =
      block_logical_coordinates(domain, interior_inertial_coords);
  for (size_t s = 0; s < expected_block_ids.size(); ++s) {
    CHECK(blogical_coords[s].value().id.get_index() == expected_block_ids[s]);
  }
  size_t element_count = 0;
  for (const auto& block : domain.blocks()) {
    const auto initial_ref_levs = shell.initial_refinement_levels();
    const std::vector<ElementId<3>> element_ids =
        initial_element_ids(block.id(), initial_ref_levs[block.id()]);
    for (const auto& element_id : element_ids) {
      const auto element = domain::Initialization::create_initial_element(
          element_id, block, initial_ref_levs);
      // Creating elements through InitialRefinement creates Elements in all
      // three logical dimensions. It is sufficient to only check the elements
      // in the radial direction lying along a ray at a fixed angle.
      // This is done by getting the desired Elements through their SegmentIds.
      if (element_id.segment_ids()[0] == SegmentId{refinement_level, 0} and
          element_id.segment_ids()[1] == SegmentId{refinement_level, 0}) {
        element_count++;
        if (block.is_time_dependent()) {
          ERROR(
              "Only stationary maps are supported in the Shell domain creator "
              "test");
        }
        const auto map = ElementMap<3, Frame::Inertial>{
            element_id, block.stationary_map().get_clone()};
        const tnsr::I<double, 3, Frame::Logical> logical_point(
            std::array<double, 3>{{0.0, 0.0, 1.0}});
        CHECK(magnitude(map(logical_point)).get() ==
              approx(x_on_element_boundary[element_count]));
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.Shell", "[Domain][Unit]") {
  test_shell_boundaries();
  test_shell_factory_equiangular();
  test_shell_factory_equidistant();
  test_shell_boundaries_aspect_ratio();
  test_shell_factory_aspect_ratio();
  test_shell_boundaries_logarithmic_map();
  test_shell_factory_logarithmic_map();
  test_shell_factory_wedges_four_on_equator();
  test_shell_factory_wedges_one_along_minus_x();
  test_shell_factory_equiangular_time_dependent();

  {
    INFO("Radial block layers 1");
    test_radial_block_layers(1.0, 10.0, 0, true, 8, {0, 1, 2, 3, 4, 5, 6, 7});
  }
  {
    INFO("Radial block layers 2");
    test_radial_block_layers(2.0, 9.0, 0, false, 8, {0, 1, 2, 3, 4, 5, 6, 7});
  }
  {
    INFO("Radial block layers 3");
    test_radial_block_layers(4.0, 9.0, 3, true, 1, {0, 0, 0, 0, 0, 0, 0, 0});
  }
  {
    INFO("Radial block layers 4");
    test_radial_block_layers(3.0, 6.0, 3, false, 1, {0, 0, 0, 0, 0, 0, 0, 0});
  }
}
}  // namespace domain
