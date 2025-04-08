// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

namespace domain {
namespace {

template <size_t Dim>
void test_block_time_independent() {
  CAPTURE(Dim);
  PUPable_reg(SINGLE_ARG(CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                       CoordinateMaps::Identity<Dim>>));

  using coordinate_map = CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                       CoordinateMaps::Identity<Dim>>;
  const coordinate_map identity_map{CoordinateMaps::Identity<Dim>{}};

  Block<Dim> original_block(identity_map.get_clone(), 7, {}, "Identity");
  CHECK_FALSE(original_block.is_time_dependent());

  const auto check_block = [](const Block<Dim>& block) {
    // Test external boundaries:
    CHECK((block.external_boundaries().size()) == 2 * Dim);

    // Test neighbors:
    CHECK((block.neighbors().size()) == 0);

    // Test id:
    CHECK((block.id()) == 7);
    CHECK(block.name() == "Identity");
    CHECK(block.topologies() == make_array<Dim>(domain::Topology::I1));

    // Test that the block's coordinate_map is Identity:
    const auto& map = block.stationary_map();
    const tnsr::I<double, Dim, Frame::BlockLogical> xi(1.0);
    const tnsr::I<double, Dim, Frame::Inertial> x(1.0);
    CHECK(map(xi) == x);
    CHECK(map.inverse(x).value() == xi);
  };

  check_block(original_block);
  check_block(serialize_and_deserialize(original_block));

  // Test PUP
  test_serialization(original_block);

  // Test move semantics:
  const Block<Dim> block_copy(identity_map.get_clone(), 7, {}, "Identity");
  test_move_semantics(std::move(original_block), block_copy);
}

template <size_t Dim>
using Translation = domain::CoordinateMaps::TimeDependent::Translation<Dim>;

template <size_t VolumeDim>
auto make_translation_map() {
  return domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
      Translation<VolumeDim>{"Translation"});
}

template <size_t VolumeDim>
auto make_translation_grid_to_distorted_map() {
  return domain::make_coordinate_map<Frame::Grid, Frame::Distorted>(
      Translation<VolumeDim>{"TranslationGridToDistorted"});
}

template <size_t VolumeDim>
auto make_translation_distorted_to_inertial_map() {
  return domain::make_coordinate_map<Frame::Distorted, Frame::Inertial>(
      Translation<VolumeDim>{"TranslationDistortedToInertial"});
}

template <size_t Dim>
void test_block_time_dependent() {
  using TranslationDimD =
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, Translation<Dim>>;
  using logical_to_grid_coordinate_map =
      CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                    CoordinateMaps::Identity<Dim>>;
  using grid_to_inertial_coordinate_map = TranslationDimD;
  PUPable_reg(logical_to_grid_coordinate_map);
  PUPable_reg(SINGLE_ARG(CoordinateMap<Frame::BlockLogical, Frame::Grid,
                                       CoordinateMaps::Identity<Dim>>));
  PUPable_reg(grid_to_inertial_coordinate_map);
  const logical_to_grid_coordinate_map identity_map{
      CoordinateMaps::Identity<Dim>{}};
  const grid_to_inertial_coordinate_map translation_map =
      make_translation_map<Dim>();
  Block<Dim> original_block(identity_map.get_clone(), 7, {});
  CHECK_FALSE(original_block.is_time_dependent());
  original_block.inject_time_dependent_map(translation_map.get_clone());
  CHECK(original_block.is_time_dependent());

  const auto check_block = [](const Block<Dim>& block) {
    const double time = 2.0;

    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        functions_of_time{};

    functions_of_time["Translation"] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
            0.0,
            std::array<DataVector, 3>{{{Dim, 0.0}, {Dim, 1.0}, {Dim, 0.0}}},
            2.5);

    // Test external boundaries:
    CHECK((block.external_boundaries().size()) == 2 * Dim);

    // Test neighbors:
    CHECK((block.neighbors().size()) == 0);

    // Test id:
    CHECK((block.id()) == 7);
    CHECK(block.topologies() == make_array<Dim>(domain::Topology::I1));

    // Test that the block's coordinate_map is Identity:
    const auto& grid_to_inertial_map = block.moving_mesh_grid_to_inertial_map();
    const auto& logical_to_grid_map = block.moving_mesh_logical_to_grid_map();
    const tnsr::I<double, Dim, Frame::BlockLogical> xi(1.0);
    const tnsr::I<double, Dim, Frame::Inertial> x(1.0 + time);
    CHECK(grid_to_inertial_map(logical_to_grid_map(xi), time,
                               functions_of_time) == x);
    CHECK(logical_to_grid_map
              .inverse(grid_to_inertial_map.inverse(x, time, functions_of_time)
                           .value())
              .value() == xi);
  };

  check_block(original_block);
  check_block(serialize_and_deserialize(original_block));

  // Test PUP
  test_serialization(original_block);

  // Test move semantics:
  Block<Dim> block_copy(identity_map.get_clone(), 7, {});
  block_copy.inject_time_dependent_map(translation_map.get_clone());
  test_move_semantics(std::move(original_block), block_copy);
}

template <size_t Dim>
void test_block_time_dependent_distorted() {
  using TranslationDimD =
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, Translation<Dim>>;
  using TranslationGridDistortedDimD =
      domain::CoordinateMap<Frame::Grid, Frame::Distorted, Translation<Dim>>;
  using TranslationDistortedInertialDimD =
      domain::CoordinateMap<Frame::Distorted, Frame::Inertial,
                            Translation<Dim>>;

  using logical_to_grid_coordinate_map =
      CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                    CoordinateMaps::Identity<Dim>>;

  using grid_to_inertial_coordinate_map = TranslationDimD;
  using grid_to_distorted_coordinate_map = TranslationGridDistortedDimD;
  using distorted_to_inertial_coordinate_map = TranslationDistortedInertialDimD;

  PUPable_reg(SINGLE_ARG(CoordinateMap<Frame::BlockLogical, Frame::Grid,
                                       CoordinateMaps::Identity<Dim>>));
  PUPable_reg(grid_to_inertial_coordinate_map);

  PUPable_reg(logical_to_grid_coordinate_map);
  PUPable_reg(SINGLE_ARG(CoordinateMap<Frame::BlockLogical, Frame::Grid,
                                       CoordinateMaps::Identity<Dim>>));
  PUPable_reg(grid_to_distorted_coordinate_map);
  PUPable_reg(distorted_to_inertial_coordinate_map);

  const logical_to_grid_coordinate_map identity_map{
      CoordinateMaps::Identity<Dim>{}};
  const grid_to_inertial_coordinate_map translation_map =
      make_translation_map<Dim>();
  const grid_to_distorted_coordinate_map translation_grid_to_distorted_map =
      make_translation_grid_to_distorted_map<Dim>();
  const distorted_to_inertial_coordinate_map
      translation_distorted_to_inertial_map =
          make_translation_distorted_to_inertial_map<Dim>();
  Block<Dim> original_block(identity_map.get_clone(), 7, {});
  CHECK_FALSE(original_block.is_time_dependent());
  original_block.inject_time_dependent_map(
      translation_map.get_clone(),
      translation_grid_to_distorted_map.get_clone(),
      translation_distorted_to_inertial_map.get_clone());
  CHECK(original_block.is_time_dependent());

  const auto check_block = [](const Block<Dim>& block) {
    const double time = 2.0;

    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        functions_of_time{};

    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        functions_of_time_grid_to_distorted{};

    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        functions_of_time_distorted_to_inertial{};

    functions_of_time["Translation"] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
            0.0,
            std::array<DataVector, 3>{{{Dim, 0.0}, {Dim, 3.0}, {Dim, 0.0}}},
            5.0);

    functions_of_time_grid_to_distorted["TranslationGridToDistorted"] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
            0.0,
            std::array<DataVector, 3>{{{Dim, 0.0}, {Dim, 1.0}, {Dim, 0.0}}},
            5.0);

    functions_of_time_distorted_to_inertial["TranslationDistortedToInertial"] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
            0.0,
            std::array<DataVector, 3>{{{Dim, 0.0}, {Dim, 2.0}, {Dim, 0.0}}},
            5.0);

    // Test external boundaries:
    CHECK((block.external_boundaries().size()) == 2 * Dim);

    // Test neighbors:
    CHECK((block.neighbors().size()) == 0);

    // Test id:
    CHECK((block.id()) == 7);
    CHECK(block.topologies() == make_array<Dim>(domain::Topology::I1));

    // Test that the block's coordinate_map is Identity:
    const auto& grid_to_inertial_map = block.moving_mesh_grid_to_inertial_map();
    const auto& grid_to_distorted_map =
        block.moving_mesh_grid_to_distorted_map();
    const auto& distorted_to_inertial_map =
        block.moving_mesh_distorted_to_inertial_map();
    const auto& logical_to_grid_map = block.moving_mesh_logical_to_grid_map();
    const tnsr::I<double, Dim, Frame::BlockLogical> xi(1.0);
    const tnsr::I<double, Dim, Frame::Inertial> x(1.0 + 3.0 * time);

    const auto& result_grid = logical_to_grid_map(xi);
    const auto& result_distorted = grid_to_distorted_map(
        result_grid, time, functions_of_time_grid_to_distorted);
    const auto& result_inertial = distorted_to_inertial_map(
        result_distorted, time, functions_of_time_distorted_to_inertial);
    CHECK(result_inertial == x);

    CHECK(logical_to_grid_map
              .inverse(grid_to_inertial_map.inverse(x, time, functions_of_time)
                           .value())
              .value() == xi);
  };

  check_block(original_block);
  check_block(serialize_and_deserialize(original_block));

  // Test PUP
  test_serialization(original_block);

  // Test move semantics:
  Block<Dim> block_copy(identity_map.get_clone(), 7, {});
  block_copy.inject_time_dependent_map(
      translation_map.get_clone(),
      translation_grid_to_distorted_map.get_clone(),
      translation_distorted_to_inertial_map.get_clone());
  test_move_semantics(std::move(original_block), block_copy);
}

void test_spherical_shell() {
  const Block<3> spherical_shell(nullptr, 4,
                                 DirectionMap<3, BlockNeighbors<3>>{}, "Shell",
                                 domain::topologies::spherical_shell);
  CHECK(spherical_shell.external_boundaries().size() == 2);
  CHECK(
      spherical_shell.external_boundaries().contains(Direction<3>::lower_xi()));
  CHECK(
      spherical_shell.external_boundaries().contains(Direction<3>::upper_xi()));
  CHECK(spherical_shell.neighbors().empty());
}

void test_cylindrical_shell() {
  const Block<3> cylindrical_shell(
      nullptr, 4, DirectionMap<3, BlockNeighbors<3>>{}, "CylindricalShell",
      domain::topologies::cylindrical_shell);
  CHECK(cylindrical_shell.external_boundaries().size() == 4);
  CHECK(cylindrical_shell.external_boundaries().contains(
      Direction<3>::lower_xi()));
  CHECK(cylindrical_shell.external_boundaries().contains(
      Direction<3>::upper_xi()));
  CHECK(cylindrical_shell.external_boundaries().contains(
      Direction<3>::lower_zeta()));
  CHECK(cylindrical_shell.external_boundaries().contains(
      Direction<3>::upper_zeta()));
  CHECK(cylindrical_shell.neighbors().empty());
}

void test_full_cylinder() {
  const Block<3> full_cylinder(nullptr, 4, DirectionMap<3, BlockNeighbors<3>>{},
                               "Cylinder", domain::topologies::full_cylinder);
  CHECK(full_cylinder.external_boundaries().size() == 3);
  CHECK(full_cylinder.external_boundaries().contains(Direction<3>::upper_xi()));
  CHECK(
      full_cylinder.external_boundaries().contains(Direction<3>::lower_zeta()));
  CHECK(
      full_cylinder.external_boundaries().contains(Direction<3>::upper_zeta()));
  CHECK(full_cylinder.neighbors().empty());
}

void test_errors() {
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ([]() {
        const BlockNeighbors<1> block_neighbors(
            1, OrientationMap<1>::create_aligned());
        const DirectionMap<1, BlockNeighbors<1>> neighbors{
            {Direction<1>::lower_xi(), block_neighbors}};
        const Block<1> loop(nullptr, 2, neighbors, "Loop",
                            std::array{domain::Topology::S1});
      }()),
      Catch::Matchers::ContainsSubstring(
          "Cannot specify a neighbor in a direction with no boundary"));
#endif
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Block", "[Domain][Unit]") {
  test_spherical_shell();
  test_cylindrical_shell();
  test_full_cylinder();
  test_errors();
  test_block_time_independent<1>();
  test_block_time_independent<2>();
  test_block_time_independent<3>();

  test_block_time_dependent<1>();
  test_block_time_dependent<2>();
  test_block_time_dependent<3>();

  test_block_time_dependent_distorted<1>();
  test_block_time_dependent_distorted<2>();
  test_block_time_dependent_distorted<3>();

  // Create DirectionMap<VolumeDim, BlockNeighbors<VolumeDim>>

  // Each BlockNeighbors is an id and an OrientationMap:
  const BlockNeighbors<2> block_neighbor1(
      1, OrientationMap<2>(std::array<Direction<2>, 2>{
             {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}));
  const BlockNeighbors<2> block_neighbor2(
      2, OrientationMap<2>(std::array<Direction<2>, 2>{
             {Direction<2>::lower_xi(), Direction<2>::upper_eta()}}));
  const DirectionMap<2, BlockNeighbors<2>> neighbors{
      {Direction<2>::upper_xi(), block_neighbor1},
      {Direction<2>::lower_eta(), block_neighbor2}};
  using coordinate_map = CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                       CoordinateMaps::Identity<2>>;
  const coordinate_map identity_map{CoordinateMaps::Identity<2>{}};
  const Block<2> block(identity_map.get_clone(), 3, neighbors, "Identity");

  // Test external boundaries:
  CHECK((block.external_boundaries().size()) == 2);

  // Test neighbors:
  CHECK((block.neighbors().size()) == 2);

  // Test id:
  CHECK((block.id()) == 3);
  CHECK(block.name() == "Identity");
  CHECK(block.topologies() == make_array<2>(domain::Topology::I1));

  // Test output:
  CHECK(get_output(block) ==
        "Block 3 (Identity):\n"
        "Topology: (I1,I1)\n"
        "Neighbors: "
        "([+0,Ids = (1); orientation = (+0, +1)],"
        "[-1,Ids = (2); orientation = (-0, +1)])\n"
        "External boundaries: (+1,-0)\n"
        "Is time dependent: false");

  // Test comparison:
  CHECK(block == block);
  {
    const Block<2> rhs(identity_map.get_clone(), 3, {}, "Identity");
    CHECK(block != rhs);
  }
  {
    const Block<2> rhs(identity_map.get_clone(), 3, neighbors, "BlockyBlock");
    CHECK(block != rhs);
  }
  {
    const Block<2> rhs(identity_map.get_clone(), 0, neighbors, "Identity");
    CHECK(block != rhs);
  }
  {
    const DirectionMap<2, BlockNeighbors<2>> annulus_neighbors{
        {Direction<2>::upper_xi(), block_neighbor1}};

    const Block<2> rhs(identity_map.get_clone(), 3, annulus_neighbors,
                       "Identity",
                       {{domain::Topology::I1, domain::Topology::S1}});
    CHECK(rhs.topologies() ==
          std::array{domain::Topology::I1, domain::Topology::S1});
    CHECK(rhs.external_boundaries().size() == 1);
    CHECK(rhs.neighbors().size() == 1);
    CHECK(block != rhs);
  }
}
}  // namespace domain
