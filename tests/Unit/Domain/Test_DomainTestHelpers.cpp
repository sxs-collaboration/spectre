// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/PolarToCartesian.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Identity1D = domain::CoordinateMaps::Identity<1>;
using Interval = domain::CoordinateMaps::Interval;
using PolarToCartesian = domain::CoordinateMaps::PolarToCartesian;

::domain::CoordinateMap<
    Frame::BlockLogical, Frame::Inertial,
    domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>>
make_affine_map_3d(const std::array<double, 3>& center,
                   const std::array<double, 3>& dimensions) {
  return domain::make_coordinate_map<Frame::BlockLogical, Frame::Inertial>(
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>{
          Affine{-1.0, 1.0, center[0] - 0.5 * dimensions[0],
                 center[0] + 0.5 * dimensions[0]},
          Affine{-1.0, 1.0, center[1] - 0.5 * dimensions[1],
                 center[1] + 0.5 * dimensions[1]},
          Affine{-1.0, 1.0, center[2] - 0.5 * dimensions[2],
                 center[2] + 0.5 * dimensions[2]}});
}

// Domain creator consisting of two cubed blocks that are conforming neighbors,
// where one block has rotated local coordinates with respect to the other.
struct ConformingCubes {
  static constexpr size_t Dim = 3;

  ConformingCubes() = default;
  Domain<Dim> create_domain() const {
    // pick a location not at the origin to make this test domain less trivial
    const std::array<double, Dim> center_block_1{-0.8, 1.3, 4.1};
    // length x width x depth
    const std::array<double, Dim> dimensions_block_1{5.0, 6.0, 7.0};

    auto coord_map_block_1 =
        make_affine_map_3d(center_block_1, dimensions_block_1);

    // block 2 has a shift in the x coord because it abuts block 1 on +x side
    const std::array<double, Dim> center_block_2{
        center_block_1[0] + dimensions_block_1[0], center_block_1[1],
        center_block_1[2]};
    const std::array<double, Dim> dimensions_block_2 = dimensions_block_1;

    // [-1, 1]^3
    auto unit_cube = make_affine_map_3d(std::array<double, Dim>{0.0, 0.0, 0.0},
                                        std::array<double, Dim>{2.0, 2.0, 2.0});
    const OrientationMap<3> rotation_block_2{std::array<Direction<Dim>, Dim>{
        Direction<3>::lower_zeta(), Direction<Dim>::lower_xi(),
        Direction<3>::upper_eta()}};
    // rotate unit cube, then translate and scale it to abut block 1
    auto coord_map_block_2 = domain::push_back(
        domain::push_back(
            unit_cube,
            domain::CoordinateMaps::DiscreteRotation<Dim>(rotation_block_2)),
        make_affine_map_3d(center_block_2, dimensions_block_2));

    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, Dim>>>
        coordinate_maps{};
    coordinate_maps.emplace_back(
        std::make_unique<std::decay_t<decltype(coord_map_block_1)>>(
            std::move(coord_map_block_1)));
    coordinate_maps.emplace_back(
        std::make_unique<std::decay_t<decltype(coord_map_block_2)>>(
            std::move(coord_map_block_2)));

    std::vector<DirectionMap<Dim, BlockNeighbors<Dim>>> block_neighbors{
        coordinate_maps.size()};
    // add block 2 as a neighbor of block 1
    block_neighbors[0].emplace(Direction<Dim>::upper_xi(),
                               BlockNeighbors<Dim>{{1},
                                                   {{1, rotation_block_2}},
                                                   /*are_conforming=*/true});
    // add block 1 as a neighbor of block 2
    block_neighbors[1].emplace(
        Direction<Dim>::upper_zeta(),
        BlockNeighbors<Dim>{{0},
                            {{0, rotation_block_2.inverse_map()}},
                            /*are_conforming=*/true});

    std::vector<Block<Dim>> blocks;
    blocks.reserve(coordinate_maps.size());

    blocks.emplace_back(std::move(coordinate_maps[0]), 0,
                        std::move(block_neighbors[0]), block_names_.at(0),
                        domain::topologies::hypercube<Dim>);
    blocks.emplace_back(std::move(coordinate_maps[1]), 1,
                        std::move(block_neighbors[1]), block_names_.at(1),
                        domain::topologies::hypercube<Dim>);

    Domain<3> domain{std::move(blocks), {}, block_groups};

    return domain;
  }

  std::vector<std::string> block_names_{"Block1", "Block2"};
  std::unordered_map<std::string, std::unordered_set<std::string>> block_groups{
      {"Blocks", {"Block1", "Block2"}}};
};

// Helper for creating a filled or hollow cylindrical block coordinate map
//
// inner_radius = 0.0 for a cylinder, inner_radius > 0.0 for a hollow cylinder
domain::CoordinateMap<
    Frame::BlockLogical, Frame::Inertial,
    domain::CoordinateMaps::ProductOf3Maps<
        ::domain::CoordinateMaps::Affine, ::domain::CoordinateMaps::Identity<1>,
        ::domain::CoordinateMaps::Interval>,
    domain::CoordinateMaps::ProductOf2Maps<
        ::domain::CoordinateMaps::PolarToCartesian,
        ::domain::CoordinateMaps::Identity<1>>>
make_cyl_coordinate_map(const double inner_radius, const double outer_radius,
                        const double lower_z_bound,
                        const double upper_z_bound) {
  const auto linear = domain::CoordinateMaps::Distribution::Linear;

  // Map: (xi, eta, zeta) in [-1,1] x [0, 2pi) x [-1, 1]
  //   xi -> r in [inner_r, outer_r]  (Affine)
  //   eta -> phi in [0, 2pi)         (Identity<1>, passes through)
  //   zeta -> z in [z_lower, z_upper] (Interval)
  // Then PolarToCartesian x Identity<1> maps (r, phi, z) -> (x, y, z)
  return domain::make_coordinate_map<Frame::BlockLogical, Frame::Inertial>(
      domain::CoordinateMaps::ProductOf3Maps<Affine, Identity1D, Interval>{
          Affine{-1.0, 1.0, inner_radius, outer_radius}, Identity1D{},
          Interval{-1.0, 1.0, lower_z_bound, upper_z_bound, linear}},
      domain::CoordinateMaps::ProductOf2Maps<PolarToCartesian, Identity1D>{
          PolarToCartesian{}, Identity1D{}});
}

// Domain creator consisting of a filled cylinder surrounded by a hollow
// cylinder, which are conforming neighbors, and where the hollow cylinder
// has its angular and axial (z) coordinates in the opposite direction of those
// of the filled cylinder.
//
// This test case is meant to target the reversal of an angular coordinate when
// we have S1 topology, because a reversal of the array of coordinates with S1
// is not symmetric like it is for the reversal of coordinates with I1. When
// using an OrientationMap to communicate that the angular coordinate of a
// conforming neighbor block is in the opposite direction of the angular
// coordinate of the host block, this lack of symmetry must be handled.
//
// More specifically, we can imagine having the filled and hollow cylinder with
// the same extents in each direction, but the hollow cylinder is flipped
// upside-down so that its angular coordinates go in the opposite direction.
// The coordinates will not geometrically overlap at this point unless we have
// an even number of coordinates, but in practice we use odd for numerical
// stability. To get the odd coordinates to overlap geometrically and have each
// block's 0th coordinate (local theta = 0) be at the same physical location,
// we could rotate the filled cylinder about its z axis by 180 degrees. Now,
// the blocks' 0th coordinates would be at the same location, but one block's
// 1st coordinate would overlap with the other's N - 1 coordinate. In this way,
// the set of coordinates would geometrically align but proceed in opposite
// local directions. While the interface here would be geometrically conforming
// and the points are seemingly aligned, data between the two blocks would not
// be communicated properly. This is because when orient_variables_on_slice()
// reverses the list of angular points, the 0th point of one cylinder will be
// mapped to the N - 1 coordinate of the other, which are not in the same
// physical location and are instead separated by 2pi/N, where N is the number
// of angular points. (This is notably different than the behavior with I1
// topology, where the 0th coordinate of one cube would be in the same physical
// location as the N - 1 coordinate.) Instead, the way to make these two
// cylinders have opposite angular directions and also for their points to be
// related by a simple OrientationMap reversal, they must be staggered by the
// additional 2pi/N. This domain creator does exactly that to make it so that
// grid points map to themselves and not their angular neighbor at the interface
// between the two blocks.
struct ConformingNestedReversedCylinders {
  static constexpr size_t Dim = 3;

  ConformingNestedReversedCylinders() = default;
  Domain<Dim> create_domain() const {
    const double inner_cyl_outer_radius = 3.0;
    const double outer_cyl_outer_radius = 5.0;
    const double height = 8.0;
    const double half_height = 0.5 * height;

    auto inner_cyl_coord_map = make_cyl_coordinate_map(
        0.0, inner_cyl_outer_radius, -half_height, half_height);

    const OrientationMap<Dim> rotate_upside_down{
        std::array<Direction<Dim>, Dim>{Direction<3>::lower_xi(),
                                        Direction<3>::upper_eta(),
                                        Direction<3>::lower_zeta()}};

    // check_block_face_grid_points_align() uses N=9 angular grid points for the
    // interface mesh, and pi - 2pi/N = 7pi/9, which will rotate the hollow
    // cylinder so that the N-1 point of the hollow cylinder overlaps with the
    // 0th point of the filled cylinder
    const double rotation_about_z = 7.0 * M_PI / 9.0;
    auto outer_cyl_coord_map = domain::push_back(
        domain::push_back(
            make_cyl_coordinate_map(inner_cyl_outer_radius,
                                    outer_cyl_outer_radius, -half_height,
                                    half_height),
            domain::CoordinateMaps::DiscreteRotation<Dim>(rotate_upside_down)),
        domain::CoordinateMaps::Rotation<3>(rotation_about_z, 0.0, 0.0));

    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, Dim>>>
        coordinate_maps{};
    coordinate_maps.emplace_back(
        std::make_unique<std::decay_t<decltype(inner_cyl_coord_map)>>(
            std::move(inner_cyl_coord_map)));
    coordinate_maps.emplace_back(
        std::make_unique<std::decay_t<decltype(outer_cyl_coord_map)>>(
            std::move(outer_cyl_coord_map)));

    const OrientationMap<Dim> inner_to_outer_cyl_map{
        std::array<Direction<Dim>, Dim>{Direction<3>::upper_xi(),
                                        Direction<3>::lower_eta(),
                                        Direction<3>::lower_zeta()}};

    std::vector<DirectionMap<Dim, BlockNeighbors<Dim>>> block_neighbors{
        coordinate_maps.size()};
    // add hollow cylinder as neighbor of filled cylinder
    block_neighbors[0].emplace(
        Direction<Dim>::upper_xi(),
        BlockNeighbors<Dim>{{1},
                            {{1, inner_to_outer_cyl_map}},
                            /*are_conforming=*/true});
    // add filled cylinder as neighbor of hollow cylinder
    block_neighbors[1].emplace(
        Direction<Dim>::lower_xi(),
        BlockNeighbors<Dim>{{0},
                            {{0, inner_to_outer_cyl_map}},
                            /*are_conforming=*/true});

    std::vector<Block<Dim>> blocks;
    blocks.reserve(coordinate_maps.size());

    blocks.emplace_back(std::move(coordinate_maps[0]), 0,
                        std::move(block_neighbors[0]), block_names_.at(0),
                        domain::topologies::full_cylinder);
    blocks.emplace_back(std::move(coordinate_maps[1]), 1,
                        std::move(block_neighbors[1]), block_names_.at(1),
                        domain::topologies::cylindrical_shell);

    Domain<3> domain{std::move(blocks), {}, block_groups};

    return domain;
  }

  std::vector<std::string> block_names_{"Block1", "Block2"};
  std::unordered_map<std::string, std::unordered_set<std::string>> block_groups{
      {"Blocks", {"Block1", "Block2"}}};
};

template <typename DataType, size_t SpatialDim>
tnsr::II<DataType, SpatialDim> random_inv_spatial_metric(
    const gsl::not_null<std::mt19937*> generator,
    const DataType& used_for_size) {
  std::uniform_real_distribution<> distribution(-0.05, 0.05);
  auto inv_spatial_metric =
      make_with_random_values<tnsr::II<DataType, SpatialDim>>(
          generator, make_not_null(&distribution), used_for_size);
  for (size_t d = 0; d < SpatialDim; ++d) {
    inv_spatial_metric.get(d, d) += 1.0;
  }
  return inv_spatial_metric;
}

template <size_t SpatialDim, typename DataType>
void test_euclidean_basis_vector(const DataType& used_for_size) {
  for (const auto& direction : Direction<SpatialDim>::all_directions()) {
    auto expected =
        make_with_value<tnsr::i<DataType, SpatialDim>>(used_for_size, 0.0);
    expected.get(direction.axis()) =
        make_with_value<DataType>(used_for_size, direction.sign());

    CHECK_ITERABLE_APPROX((euclidean_basis_vector(direction, used_for_size)),
                          std::move(expected));
  }
}

template <size_t SpatialDim, typename DataType>
void test_unit_basis_form(const DataType& used_for_size) {
  MAKE_GENERATOR(generator);
  const auto inv_spatial_metric =
      random_inv_spatial_metric<DataType, SpatialDim>(make_not_null(&generator),
                                                      used_for_size);
  for (const auto& direction : Direction<SpatialDim>::all_directions()) {
    const auto basis_form = unit_basis_form(direction, inv_spatial_metric);
    auto expected = euclidean_basis_vector(direction, used_for_size);
    const DataType norm = get(magnitude(expected, inv_spatial_metric));
    for (size_t d = 0; d < SpatialDim; ++d) {
      expected.get(d) /= norm;
    }
    CHECK_ITERABLE_APPROX(basis_form, expected);
    CHECK_ITERABLE_APPROX(get(magnitude(expected, inv_spatial_metric)),
                          make_with_value<DataType>(used_for_size, 1.0));
  }
}

// Test the domain test helper, test_physical_separation()
void test_test_physical_separation() {
  // Test conforming interface with I1 topology and rotated local coordinates
  const ConformingCubes conforming_cubes_creator{};
  const Domain<3> conforming_cubes_domain =
      conforming_cubes_creator.create_domain();
  const auto& conforming_cubes_blocks = conforming_cubes_domain.blocks();
  test_physical_separation(conforming_cubes_blocks, 0.0);

  // Test conforming interface with S1 topology and reversed local coordinates
  const ConformingNestedReversedCylinders
      conforming_nested_reversed_cylinders_creator{};
  const Domain<3> conforming_nested_reversed_cylinders_domain =
      conforming_nested_reversed_cylinders_creator.create_domain();
  const auto& conforming_nested_reversed_cylinders_blocks =
      conforming_nested_reversed_cylinders_domain.blocks();
  test_physical_separation(conforming_nested_reversed_cylinders_blocks, 0.0);
}
}  //  namespace

SPECTRE_TEST_CASE("Unit.Domain.TestHelpers", "[Unit][Domain]") {
  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_euclidean_basis_vector, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_unit_basis_form, (1, 2, 3));

  test_test_physical_separation();
}
