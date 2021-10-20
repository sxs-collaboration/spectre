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
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/GetOutput.hpp"

namespace domain {
namespace {
namespace helpers = ::TestHelpers::domain::BoundaryConditions;

template <size_t Dim>
auto create_external_bcs() {
  DirectionMap<Dim,
               std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>
      external_boundary_conditions{};
  for (const auto& direction : Direction<Dim>::all_directions()) {
    external_boundary_conditions[direction] =
        std::make_unique<helpers::TestBoundaryCondition<Dim>>(direction);
  }
  return external_boundary_conditions;
}

template <size_t Dim>
void test_block_time_independent() {
  CAPTURE(Dim);
  PUPable_reg(SINGLE_ARG(CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                       CoordinateMaps::Identity<Dim>>));

  using coordinate_map = CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                       CoordinateMaps::Identity<Dim>>;
  const coordinate_map identity_map{CoordinateMaps::Identity<Dim>{}};

  Block<Dim> original_block(identity_map.get_clone(), 7, {},
                            create_external_bcs<Dim>());
  CHECK_FALSE(original_block.is_time_dependent());

  const auto check_block = [](const Block<Dim>& block) {
    // Test external boundaries:
    CHECK((block.external_boundaries().size()) == 2 * Dim);

    // Test neighbors:
    CHECK((block.neighbors().size()) == 0);

    // Test id:
    CHECK((block.id()) == 7);

    // Test that the block's coordinate_map is Identity:
    const auto& map = block.stationary_map();
    const tnsr::I<double, Dim, Frame::BlockLogical> xi(1.0);
    const tnsr::I<double, Dim, Frame::Inertial> x(1.0);
    CHECK(map(xi) == x);
    CHECK(map.inverse(x).value() == xi);

    for (const auto& direction : Direction<Dim>::all_directions()) {
      CAPTURE(direction);
      REQUIRE(block.external_boundary_conditions().at(direction) != nullptr);
      CHECK(dynamic_cast<const helpers::TestBoundaryCondition<Dim>&>(
                *block.external_boundary_conditions().at(direction))
                .direction() == direction);
    }
  };

  check_block(original_block);
  check_block(serialize_and_deserialize(original_block));

  // Test PUP
  test_serialization(original_block);

  // Test move semantics:
  const Block<Dim> block_copy(identity_map.get_clone(), 7, {},
                              create_external_bcs<Dim>());
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
  Block<Dim> original_block(identity_map.get_clone(), 7, {},
                            create_external_bcs<Dim>());
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

    for (const auto& direction : Direction<Dim>::all_directions()) {
      CAPTURE(direction);
      REQUIRE(block.external_boundary_conditions().at(direction) != nullptr);
      CHECK(dynamic_cast<const helpers::TestBoundaryCondition<Dim>&>(
                *block.external_boundary_conditions().at(direction))
                .direction() == direction);
    }
  };

  check_block(original_block);
  check_block(serialize_and_deserialize(original_block));

  // Test PUP
  test_serialization(original_block);

  // Test move semantics:
  Block<Dim> block_copy(identity_map.get_clone(), 7, {},
                        create_external_bcs<Dim>());
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
  Block<Dim> original_block(identity_map.get_clone(), 7, {},
                            create_external_bcs<Dim>());
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

    for (const auto& direction : Direction<Dim>::all_directions()) {
      CAPTURE(direction);
      REQUIRE(block.external_boundary_conditions().at(direction) != nullptr);
      CHECK(dynamic_cast<const helpers::TestBoundaryCondition<Dim>&>(
                *block.external_boundary_conditions().at(direction))
                .direction() == direction);
    }
  };

  check_block(original_block);
  check_block(serialize_and_deserialize(original_block));

  // Test PUP
  test_serialization(original_block);

  // Test move semantics:
  Block<Dim> block_copy(identity_map.get_clone(), 7, {},
                        create_external_bcs<Dim>());
  block_copy.inject_time_dependent_map(
      translation_map.get_clone(),
      translation_grid_to_distorted_map.get_clone(),
      translation_distorted_to_inertial_map.get_clone());
  test_move_semantics(std::move(original_block), block_copy);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Block", "[Domain][Unit]") {
  helpers::register_derived_with_charm();

  test_block_time_independent<1>();
  test_block_time_independent<2>();
  test_block_time_independent<3>();

  test_block_time_dependent<1>();
  test_block_time_dependent<2>();
  test_block_time_dependent<3>();

  test_block_time_dependent_distorted<1>();
  test_block_time_dependent_distorted<2>();
  test_block_time_dependent_distorted<3>();

  // Create DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>

  // Each BlockNeighbor is an id and an OrientationMap:
  const BlockNeighbor<2> block_neighbor1(
      1, OrientationMap<2>(std::array<Direction<2>, 2>{
             {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}));
  const BlockNeighbor<2> block_neighbor2(
      2, OrientationMap<2>(std::array<Direction<2>, 2>{
             {Direction<2>::lower_xi(), Direction<2>::upper_eta()}}));
  DirectionMap<2, BlockNeighbor<2>> neighbors = {
      {Direction<2>::upper_xi(), block_neighbor1},
      {Direction<2>::lower_eta(), block_neighbor2}};
  using coordinate_map = CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                       CoordinateMaps::Identity<2>>;
  const coordinate_map identity_map{CoordinateMaps::Identity<2>{}};
  auto external_boundary_conditions = create_external_bcs<2>();
  external_boundary_conditions.erase(Direction<2>::upper_xi());
  external_boundary_conditions.erase(Direction<2>::lower_eta());
  const Block<2> block(identity_map.get_clone(), 3, std::move(neighbors),
                       std::move(external_boundary_conditions));

  // Test external boundaries:
  CHECK((block.external_boundaries().size()) == 2);

  // Test neighbors:
  CHECK((block.neighbors().size()) == 2);

  // Test id:
  CHECK((block.id()) == 3);

  // Test output:
  CHECK(get_output(block) ==
        "Block 3:\n"
        "Neighbors: "
        "([+0,Id = 1; orientation = (+0, +1)],"
        "[-1,Id = 2; orientation = (-0, +1)])\n"
        "External boundaries: (+1,-0)\n"
        "Is time dependent: false");

  // Test comparison:
  const Block<2> neighborless_block(identity_map.get_clone(), 7, {});
  CHECK(block == block);
  CHECK(block != neighborless_block);
  CHECK(neighborless_block == neighborless_block);
}
}  // namespace domain
