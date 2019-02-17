// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/IdPair.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/BlockId.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/BlockNeighbor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/EquatorialCompression.hpp"
#include "Domain/CoordinateMaps/Wedge3D.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/SegmentId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/TestCreation.hpp"

// IWYU pragma: no_forward_declare BlockNeighbor

namespace domain {
namespace {
void test_shell_construction(
    const creators::Shell<Frame::Inertial>& shell, const double inner_radius,
    const double outer_radius, const bool use_equiangular_map,
    const std::array<size_t, 2>& expected_shell_extents,
    const std::vector<std::array<size_t, 3>>& expected_refinement_level,
    const double aspect_ratio = 1.0, const bool use_logarithmic_map = false,
    const ShellWedges which_wedges = ShellWedges::All) {
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
  using Wedge3DMap = CoordinateMaps::Wedge3D;
  using Halves = Wedge3DMap::WedgeHalves;
  if (aspect_ratio == 1.0) {
    auto vector_of_maps = make_vector_coordinate_map_base<Frame::Logical,
                                                          Frame::Inertial>(
        Wedge3DMap{inner_radius, outer_radius, OrientationMap<3>{}, 1.0, 1.0,
                   use_equiangular_map, Halves::Both, use_logarithmic_map},
        Wedge3DMap{inner_radius, outer_radius,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                        Direction<3>::lower_zeta()}}},
                   1.0, 1.0, use_equiangular_map, Halves::Both,
                   use_logarithmic_map},
        Wedge3DMap{inner_radius, outer_radius,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
                        Direction<3>::lower_eta()}}},
                   1.0, 1.0, use_equiangular_map, Halves::Both,
                   use_logarithmic_map},
        Wedge3DMap{inner_radius, outer_radius,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
                        Direction<3>::upper_eta()}}},
                   1.0, 1.0, use_equiangular_map, Halves::Both,
                   use_logarithmic_map},
        Wedge3DMap{inner_radius, outer_radius,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_zeta(), Direction<3>::upper_xi(),
                        Direction<3>::upper_eta()}}},
                   1.0, 1.0, use_equiangular_map, Halves::Both,
                   use_logarithmic_map},
        Wedge3DMap{inner_radius, outer_radius,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::lower_zeta(), Direction<3>::lower_xi(),
                        Direction<3>::upper_eta()}}},
                   1.0, 1.0, use_equiangular_map, Halves::Both,
                   use_logarithmic_map});
    if (UNLIKELY(which_wedges == ShellWedges::FourOnEquator)) {
      vector_of_maps.erase(vector_of_maps.begin(), vector_of_maps.begin() + 2);
    } else if (UNLIKELY(which_wedges == ShellWedges::OneAlongMinusX)) {
      vector_of_maps.erase(vector_of_maps.begin(), vector_of_maps.begin() + 5);
    }
    test_domain_construction(domain, expected_block_neighbors,
                             expected_external_boundaries, vector_of_maps);
  } else {
    const auto compression =
        CoordinateMaps::EquatorialCompression{aspect_ratio};
    auto vector_of_maps = make_vector(
        make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            Wedge3DMap{inner_radius, outer_radius, OrientationMap<3>{}, 1.0,
                       1.0, use_equiangular_map, Halves::Both,
                       use_logarithmic_map},
            compression),
        make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            Wedge3DMap{inner_radius, outer_radius,
                       OrientationMap<3>{std::array<Direction<3>, 3>{
                           {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                            Direction<3>::lower_zeta()}}},
                       1.0, 1.0, use_equiangular_map, Halves::Both,
                       use_logarithmic_map},
            compression),
        make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            Wedge3DMap{
                inner_radius, outer_radius,
                OrientationMap<3>{std::array<Direction<3>, 3>{
                    {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
                     Direction<3>::lower_eta()}}},
                1.0, 1.0, use_equiangular_map, Halves::Both,
                use_logarithmic_map},
            compression),
        make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            Wedge3DMap{
                inner_radius, outer_radius,
                OrientationMap<3>{std::array<Direction<3>, 3>{
                    {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
                     Direction<3>::upper_eta()}}},
                1.0, 1.0, use_equiangular_map, Halves::Both,
                use_logarithmic_map},
            compression),
        make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            Wedge3DMap{
                inner_radius, outer_radius,
                OrientationMap<3>{std::array<Direction<3>, 3>{
                    {Direction<3>::upper_zeta(), Direction<3>::upper_xi(),
                     Direction<3>::upper_eta()}}},
                1.0, 1.0, use_equiangular_map, Halves::Both,
                use_logarithmic_map},
            compression),
        make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            Wedge3DMap{
                inner_radius, outer_radius,
                OrientationMap<3>{std::array<Direction<3>, 3>{
                    {Direction<3>::lower_zeta(), Direction<3>::lower_xi(),
                     Direction<3>::upper_eta()}}},
                1.0, 1.0, use_equiangular_map, Halves::Both,
                use_logarithmic_map},
            compression));
    if (UNLIKELY(which_wedges == ShellWedges::FourOnEquator)) {
      vector_of_maps.erase(vector_of_maps.begin(), vector_of_maps.begin() + 2);
    } else if (UNLIKELY(which_wedges == ShellWedges::OneAlongMinusX)) {
      vector_of_maps.erase(vector_of_maps.begin(), vector_of_maps.begin() + 5);
    }
    test_domain_construction(domain, expected_block_neighbors,
                             expected_external_boundaries, vector_of_maps);
  }

  test_initial_domain(domain, shell.initial_refinement_levels());
}

void test_shell_boundaries() {
  INFO("Shell boundaries");
  const double inner_radius = 1.0, outer_radius = 2.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points_r_angular{{4, 4}};

  for (const auto& use_equiangular_map : {true, false}) {
    const creators::Shell<Frame::Inertial> shell{
        inner_radius, outer_radius, refinement_level, grid_points_r_angular,
        use_equiangular_map};
    test_physical_separation(shell.create_domain().blocks());
    test_shell_construction(shell, inner_radius, outer_radius,
                            use_equiangular_map, grid_points_r_angular,
                            {6, make_array<3>(refinement_level)});
  }
}

void test_shell_factory_equiangular() {
  INFO("Shell factory equiangular");
  const auto shell = test_factory_creation<DomainCreator<3, Frame::Inertial>>(
      "  Shell:\n"
      "    InnerRadius: 1\n"
      "    OuterRadius: 3\n"
      "    InitialRefinement: 2\n"
      "    InitialGridPoints: [2,3]\n");
  const double inner_radius = 1.0, outer_radius = 3.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
  test_shell_construction(
      dynamic_cast<const creators::Shell<Frame::Inertial>&>(*shell),
      inner_radius, outer_radius, true, grid_points_r_angular,
      {6, make_array<3>(refinement_level)});
}

void test_shell_factory_equidistant() {
  INFO("Shell factory equidistant");
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
      dynamic_cast<const creators::Shell<Frame::Inertial>&>(*shell),
      inner_radius, outer_radius, false, grid_points_r_angular,
      {6, make_array<3>(refinement_level)});
}

void test_shell_boundaries_aspect_ratio() {
  INFO("Shell boundaries aspect ratio");
  const double inner_radius = 1.0, outer_radius = 2.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points_r_angular{{4, 4}};
  const double aspect_ratio = 1.3;

  const creators::Shell<Frame::Inertial> shell{
      inner_radius,          outer_radius, refinement_level,
      grid_points_r_angular, false,        aspect_ratio};
  test_physical_separation(shell.create_domain().blocks());
  test_shell_construction(shell, inner_radius, outer_radius, false,
                          grid_points_r_angular,
                          {6, make_array<3>(refinement_level)}, aspect_ratio);
}

void test_shell_factory_aspect_ratio() {
  INFO("Shell factory aspect ratio");
  const auto shell = test_factory_creation<DomainCreator<3, Frame::Inertial>>(
      "  Shell:\n"
      "    InnerRadius: 1\n"
      "    OuterRadius: 3\n"
      "    InitialRefinement: 2\n"
      "    InitialGridPoints: [2,3]\n"
      "    UseEquiangularMap: false\n"
      "    AspectRatio: 2.0        \n");
  const double inner_radius = 1.0, outer_radius = 3.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
  const double aspect_ratio = 2.0;
  test_shell_construction(
      dynamic_cast<const creators::Shell<Frame::Inertial>&>(*shell),
      inner_radius, outer_radius, false, grid_points_r_angular,
      {6, make_array<3>(refinement_level)}, aspect_ratio);
}

void test_shell_boundaries_logarithmic_map() {
  INFO("Shell boundaries logarithmic map");
  const double inner_radius = 1.0, outer_radius = 2.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points_r_angular{{4, 4}};
  const double aspect_ratio = 1.0;
  const bool use_logarithmic_map = true;

  const creators::Shell<Frame::Inertial> shell{
      inner_radius, outer_radius, refinement_level,   grid_points_r_angular,
      false,        aspect_ratio, use_logarithmic_map};
  test_physical_separation(shell.create_domain().blocks());
  test_shell_construction(
      shell, inner_radius, outer_radius, false, grid_points_r_angular,
      {6, make_array<3>(refinement_level)}, aspect_ratio, use_logarithmic_map);
}

void test_shell_factory_logarithmic_map() {
  INFO("Shell factory logarithmic map");
  const auto shell = test_factory_creation<DomainCreator<3, Frame::Inertial>>(
      "  Shell:\n"
      "    InnerRadius: 1\n"
      "    OuterRadius: 3\n"
      "    InitialRefinement: 2\n"
      "    InitialGridPoints: [2,3]\n"
      "    UseEquiangularMap: false\n"
      "    AspectRatio: 2.0        \n"
      "    UseLogarithmicMap: true\n");
  const double inner_radius = 1.0, outer_radius = 3.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
  const double aspect_ratio = 2.0;
  const bool use_logarithmic_map = true;
  test_shell_construction(
      dynamic_cast<const creators::Shell<Frame::Inertial>&>(*shell),
      inner_radius, outer_radius, false, grid_points_r_angular,
      {6, make_array<3>(refinement_level)}, aspect_ratio, use_logarithmic_map);
}

void test_shell_factory_wedges_four_on_equator() {
  INFO("Shell factory wedges four on equator");
  const auto shell = test_factory_creation<DomainCreator<3, Frame::Inertial>>(
      "  Shell:\n"
      "    InnerRadius: 1\n"
      "    OuterRadius: 3\n"
      "    InitialRefinement: 2\n"
      "    InitialGridPoints: [2,3]\n"
      "    UseEquiangularMap: false\n"
      "    AspectRatio: 2.0        \n"
      "    UseLogarithmicMap: true\n"
      "    WhichWedges: FourOnEquator\n");
  const double inner_radius = 1.0, outer_radius = 3.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
  const double aspect_ratio = 2.0;
  const bool use_logarithmic_map = true;
  const ShellWedges which_wedges = ShellWedges::FourOnEquator;
  test_shell_construction(
      dynamic_cast<const creators::Shell<Frame::Inertial>&>(*shell),
      inner_radius, outer_radius, false, grid_points_r_angular,
      {4, make_array<3>(refinement_level)}, aspect_ratio, use_logarithmic_map,
      which_wedges);
}

void test_shell_factory_wedges_one_along_minus_x() {
  INFO("Shell factory wedges one along minus x");
  const auto shell = test_factory_creation<DomainCreator<3, Frame::Inertial>>(
      "  Shell:\n"
      "    InnerRadius: 2\n"
      "    OuterRadius: 3\n"
      "    InitialRefinement: 2\n"
      "    InitialGridPoints: [2,3]\n"
      "    UseEquiangularMap: true\n"
      "    AspectRatio: 2.7        \n"
      "    UseLogarithmicMap: false\n"
      "    WhichWedges: OneAlongMinusX \n");
  const double inner_radius = 2.0, outer_radius = 3.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
  const double aspect_ratio = 2.7;
  const bool use_logarithmic_map = false;
  const ShellWedges which_wedges = ShellWedges::OneAlongMinusX;
  test_shell_construction(
      dynamic_cast<const creators::Shell<Frame::Inertial>&>(*shell),
      inner_radius, outer_radius, true, grid_points_r_angular,
      {1, make_array<3>(refinement_level)}, aspect_ratio, use_logarithmic_map,
      which_wedges);
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

  const auto zero = make_with_value<DataVector>(x_in_block_interior, 0.0);
  tnsr::I<DataVector, 3, Frame::Inertial> interior_inertial_coords{
      {{-x_in_block_interior, zero, zero}}};
  const creators::Shell<Frame::Inertial> shell{
      inner_radius,          outer_radius,        refinement_level,
      grid_points_r_angular, use_equiangular_map, aspect_ratio,
      use_logarithmic_map,   which_wedges,        radial_block_layers};
  auto domain = shell.create_domain();
  const auto blogical_coords =
      block_logical_coordinates(domain, interior_inertial_coords);
  for (size_t s = 0; s < expected_block_ids.size(); ++s) {
    CHECK(blogical_coords[s].id.get_index() == expected_block_ids[s]);
  }
  size_t element_count = 0;
  for (const auto& block : domain.blocks()) {
    const auto initial_ref_levs = shell.initial_refinement_levels()[block.id()];
    const std::vector<ElementId<3>> element_ids =
        initial_element_ids(block.id(), initial_ref_levs);
    for (const auto& element_id : element_ids) {
      const auto element = create_initial_element(element_id, block);
      // Creating elements through InitialRefinement creates Elements in all
      // three logical dimensions. It is sufficient to only check the elements
      // in the radial direction lying along a ray at a fixed angle.
      // This is done by getting the desired Elements through their SegmentIds.
      if (element_id.segment_ids()[0] == SegmentId{refinement_level, 0} and
          element_id.segment_ids()[1] == SegmentId{refinement_level, 0}) {
        element_count++;
        const auto map = ElementMap<3, Frame::Inertial>{
            element_id, block.coordinate_map().get_clone()};
        const tnsr::I<double, 3, Frame::Logical> logical_point(
            std::array<double, 3>{{0.0, 0.0, 1.0}});
        CHECK(magnitude(map(logical_point)).get() ==
              approx(x_on_element_boundary[element_count]));
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.Shell", "[Domain][Unit]") {
  test_shell_boundaries();
  test_shell_factory_equiangular();
  test_shell_factory_equidistant();
  test_shell_boundaries_aspect_ratio();
  test_shell_factory_aspect_ratio();
  test_shell_boundaries_logarithmic_map();
  test_shell_factory_logarithmic_map();
  test_shell_factory_wedges_four_on_equator();
  test_shell_factory_wedges_one_along_minus_x();

  {
    INFO("shell factory logarithmic block layers");
    const auto log_shell =
        test_factory_creation<DomainCreator<3, Frame::Inertial>>(
            "  Shell:\n"
            "    InnerRadius: 1\n"
            "    OuterRadius: 3\n"
            "    InitialRefinement: 2\n"
            "    InitialGridPoints: [2,3]\n"
            "    UseEquiangularMap: false\n"
            "    AspectRatio: 2.0        \n"
            "    UseLogarithmicMap: true\n"
            "    RadialBlockLayers: 6\n");
  }
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
