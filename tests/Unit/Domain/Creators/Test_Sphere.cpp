// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/BulgedCube.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/EquatorialCompression.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain {
namespace {
using Translation3D = CoordinateMaps::TimeDependent::Translation<3>;

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_boundary_condition() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::upper_zeta(), 50);
}

// Calculate block logical coordinates residing on corners of
// inner block or block faces of wedges
// whose normal vector point radially outward from the
// origin.  With this domain, this direction corresponds to
// upper zeta.  These coordinates will be used to ensure
// they lie on concentric spheres defined by either the inner
// sphere, outer sphere, or radial partition parameters.
tnsr::I<double, 3, Frame::BlockLogical> logical_coords(
    const gsl::not_null<std::mt19937*> generator, const size_t num_blocks,
    const size_t block_id, const bool abuts_inner_block) {
  std::uniform_real_distribution<> real_dis(-1, 1);

  const double rand_int_xi = (2.0 * (rand() % 2) - 1.0);
  const double rand_int_eta = (2.0 * (rand() % 2) - 1.0);
  const double rand_int_zeta = (2.0 * (rand() % 2) - 1.0);
  const double rand_real_xi = real_dis(*generator);
  const double rand_real_eta = real_dis(*generator);

  double xi_logical_coord;
  double eta_logical_coord;
  // enforce coordinates either fall on the lower or
  // upper zeta face of wedges
  const double zeta_logical_coord = rand_int_zeta;

  if (block_id == num_blocks - 1) {
    // inner block only uses integer corners
    xi_logical_coord = rand_int_xi;
    eta_logical_coord = rand_int_eta;

  } else if (abuts_inner_block) {
    // next to inner block,
    // corners only on lower face b/c of square inner block neighbor
    // face
    xi_logical_coord = rand_int_xi;
    eta_logical_coord = rand_int_eta;

    // anywhere on upper zeta face b/c adjacent with spherical wedge
    if (rand_int_zeta == 1) {
      xi_logical_coord = rand_real_xi;
      eta_logical_coord = rand_real_eta;
    }
  } else {
    // adjacent to wedges
    // everywhere on low or high face should lie on a sphere b/c
    // neighbor with spherical wedge
    xi_logical_coord = rand_real_xi;
    eta_logical_coord = rand_real_eta;
  }

  return tnsr::I<double, 3, Frame::BlockLogical>{
      {{xi_logical_coord, eta_logical_coord, zeta_logical_coord}}};
}

void test_sphere_construction(
    const creators::Sphere& sphere, const double inner_radius,
    const double outer_radius, const std::vector<double>& radial_partitioning,
    const bool expect_boundary_conditions,
    const std::vector<double>& times = {0.},
    const std::array<double, 3>& velocity = {{0., 0., 0.}}) {
  // check consistency of domain
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      sphere, expect_boundary_conditions, false, times);

  const auto& blocks = domain.blocks();
  const auto block_names = sphere.block_names();
  const size_t num_blocks = blocks.size();
  const auto all_boundary_conditions = sphere.external_boundary_conditions();
  const auto functions_of_time = sphere.functions_of_time();

  // construct vector of inner radius, outer radius, and refinements levels
  // where inertial block corners have to be located
  std::vector<double> expected_corner_radii = radial_partitioning;
  expected_corner_radii.insert(expected_corner_radii.begin(), inner_radius);
  expected_corner_radii.emplace_back(outer_radius);

  MAKE_GENERATOR(generator);

  // verify if adjacent to inner block
  const auto abuts_inner_block =
      [&num_blocks](const auto& direction_and_neighbor) {
        return direction_and_neighbor.second.id() == num_blocks - 1;
      };

  for (size_t block_id = 0; block_id < num_blocks; ++block_id) {
    CAPTURE(block_id);
    const auto& block = blocks[block_id];
    const auto& boundary_conditions = all_boundary_conditions[block_id];

    {
      INFO("Block boundaries are spherical");
      // This section tests if the logical coordinates of corners from all
      // blocks (and points on upper wedge faces) lie on spherical shells
      // specified by inner radius, radial partitions, or outer radius
      const auto coords_on_spherical_partition =
          logical_coords(make_not_null(&generator), num_blocks, block_id,
                         alg::any_of(block.neighbors(), abuts_inner_block));
      for (const double current_time : times) {
        CAPTURE(current_time);
        const double corner_distance_from_origin =
            [&block, &coords_on_spherical_partition, &current_time,
             &functions_of_time, &velocity]() -> double {
          // use stationary map if independent of time
          if (not block.is_time_dependent()) {
            return get(magnitude(
                block.stationary_map()(coords_on_spherical_partition)));
          } else {
            // go from logical to grid coords, then grid to inertial coords
            const auto inertial_location_time_dep =
                block.moving_mesh_grid_to_inertial_map()(
                    block.moving_mesh_logical_to_grid_map()(
                        coords_on_spherical_partition, current_time,
                        functions_of_time),
                    current_time, functions_of_time);
            // origin in inertial frame (need to shift inertial
            // coord by velocity * (final_time - initial_time))
            return sqrt(square(get<0>(inertial_location_time_dep) -
                               velocity[0] * (current_time - 1.0)) +
                        square(get<1>(inertial_location_time_dep) -
                               velocity[1] * (current_time - 1.0)) +
                        square(get<2>(inertial_location_time_dep) -
                               velocity[2] * (current_time - 1.0)));
          }   // end time-dependent if/else
        }();  // end lambda
        CAPTURE(corner_distance_from_origin);
        const auto match_demarcation =
            [&corner_distance_from_origin](const double radius) {
              return corner_distance_from_origin == approx(radius);
            };
        CHECK(alg::any_of(expected_corner_radii, match_demarcation));
      }
    }

    // if block has 5 neighbors, 1 face should be external, and that direction
    // should be upper_zeta, for the sphere
    if (block.neighbors().size() == 5) {
      CHECK(size(block.external_boundaries()) == 1);
      CHECK(*begin(block.external_boundaries()) == Direction<3>::upper_zeta());

      // also 5 neighbor blocks should only have 1 boundary condition
      if (expect_boundary_conditions) {
        CHECK(size(boundary_conditions) == 1);
      }
      // Consistency check for like neighbor block directions: if block has 5
      // neighbors, it's external --> four of the neighbors should have upper
      // zeta external boundaries
      size_t neighbor_count = 0;
      for (auto neighbor : block.neighbors()) {
        auto neighbor_id = neighbor.second;

        if (size(blocks[neighbor_id.id()].external_boundaries()) == 1) {
          if (*begin(blocks[neighbor_id.id()].external_boundaries()) ==
              Direction<3>::upper_zeta()) {
            neighbor_count++;
          }
        }
      }

      CHECK(neighbor_count == 4);
      // if > 5 neighbors, none should have external boundaries
    } else if (block.neighbors().size() == 6) {
      // internal block case
      CHECK(size(block.external_boundaries()) == 0);
      // internal blocks should not have boundary conditions
      if (expect_boundary_conditions) {
        CHECK(size(boundary_conditions) == 0);
      }
    } else {
      // If here, something is wrong; should only have 5 or 6 neighbors, so
      // throw a guaranteed failure.
      const bool block_does_not_have_correct_number_of_neighbors = false;
      CHECK(block_does_not_have_correct_number_of_neighbors);
    }
  }  // block loop
}  // test_sphere_construction()

// ensure CHECK_THROWS_WITH calls are properly captured
void test_parse_errors() {
  INFO("Sphere check throws");
  const double inner_radius = 1.0;
  const double outer_radius = 2.0;
  const size_t refinement = 2;
  const std::array<size_t, 3> initial_extents{{4, 5, 6}};
  const std::vector<double> radial_partitioning = {};
  const std::vector<double> radial_partitioning_unordered = {
      {1.5 * inner_radius, 1.1 * inner_radius}};
  const std::vector<double> radial_partitioning_low = {
      {0.5 * inner_radius, 1.1 * inner_radius}};
  const std::vector<double> radial_partitioning_high = {
      {2.1 * outer_radius, 2.2 * outer_radius}};
  const std::vector<domain::CoordinateMaps::Distribution> radial_distribution{
      domain::CoordinateMaps::Distribution::Linear};
  const std::vector<domain::CoordinateMaps::Distribution>
      radial_distribution_too_many{
          domain::CoordinateMaps::Distribution::Linear,
          domain::CoordinateMaps::Distribution::Logarithmic};
  const std::vector<domain::CoordinateMaps::Distribution>
      radial_distribution_inner_log{
          domain::CoordinateMaps::Distribution::Logarithmic};

  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, outer_radius, -1.0, refinement, initial_extents, true,
          radial_partitioning, radial_distribution, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Inner cube sphericity must be >= 0.0 and strictly < 1.0"));
  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, outer_radius, 1.0, refinement, initial_extents, true,
          radial_partitioning, radial_distribution, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Inner cube sphericity must be >= 0.0 and strictly < 1.0"));
  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, outer_radius, 2.0, refinement, initial_extents, true,
          radial_partitioning, radial_distribution, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Inner cube sphericity must be >= 0.0 and strictly < 1.0"));

  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, 0.5 * inner_radius, 0.5, refinement, initial_extents,
          true, radial_partitioning, radial_distribution, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Inner radius must be smaller than outer radius"));

  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, outer_radius, 0.5, refinement, initial_extents, true,
          radial_partitioning_unordered, radial_distribution, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Specify radial partitioning in ascending order."));

  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, outer_radius, 0.5, refinement, initial_extents, true,
          radial_partitioning_low, radial_distribution, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "First radial partition must be larger than inner"));
  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, outer_radius, 0.5, refinement, initial_extents, true,
          radial_partitioning_high, radial_distribution, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Last radial partition must be smaller than outer"));
  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, outer_radius, 0.5, refinement, initial_extents, true,
          radial_partitioning, radial_distribution_too_many, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Specify a 'RadialDistribution' for every spherical shell. You"));
  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, outer_radius, 0.5, refinement, initial_extents, true,
          radial_partitioning, radial_distribution_inner_log, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "The 'RadialDistribution' must be 'Linear' for the"));

  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, outer_radius, 0.0, refinement, initial_extents, true,
          radial_partitioning, radial_distribution, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Cannot have periodic boundary conditions with a Sphere"));
  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, outer_radius, 0.0, refinement, initial_extents, true,
          radial_partitioning, radial_distribution, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "None boundary condition is not supported. If you would like "
          "an outflow-type boundary condition, you must use that."));
}

// Check wedge neighbors have consistent directions & orientations
void test_sphere_boundaries() {
  INFO("Ensure sphere boundaries are equidistant");
  const double inner_radius = 1.0;
  const double outer_radius = 2.0;
  const size_t initial_refinement = 3;
  const std::array<size_t, 3> initial_extents{{4, 5, 6}};
  const std::array<std::vector<double>, 2> radial_partitioning{
      {{}, {0.5 * (inner_radius + outer_radius)}}};
  const std::array<std::vector<domain::CoordinateMaps::Distribution>, 2>
      radial_distribution{
          {{domain::CoordinateMaps::Distribution::Linear},
           {domain::CoordinateMaps::Distribution::Linear,
            domain::CoordinateMaps::Distribution::Logarithmic}}};

  for (const auto& [sphericity, equiangular, array_index] :
       cartesian_product(make_array(0.0, 0.7), make_array(false, true),
                         make_array(0.0, 1.0))) {
    CAPTURE(inner_radius);
    CAPTURE(outer_radius);
    CAPTURE(sphericity);
    CAPTURE(initial_refinement);
    CAPTURE(initial_extents);
    CAPTURE(equiangular);
    CAPTURE(radial_partitioning[array_index]);
    CAPTURE(radial_distribution[array_index]);

    const creators::Sphere sphere{inner_radius,
                                  outer_radius,
                                  sphericity,
                                  initial_refinement,
                                  initial_extents,
                                  equiangular,
                                  radial_partitioning[array_index],
                                  radial_distribution[array_index]};

    test_sphere_construction(sphere, inner_radius, outer_radius,
                             radial_partitioning[array_index], false);

    const creators::Sphere sphere_boundary_condition{
        inner_radius,
        outer_radius,
        sphericity,
        initial_refinement,
        initial_extents,
        equiangular,
        radial_partitioning[array_index],
        radial_distribution[array_index],
        nullptr,
        create_boundary_condition()};

    test_sphere_construction(sphere_boundary_condition, inner_radius,
                             outer_radius, radial_partitioning[array_index],
                             true);
  }
}

// Check wedge neighbors have consistent directions & orientations, with time
// dependence
void test_sphere_factory_time_dependent() {
  INFO("Sphere factory time dependent");
  const auto domain_creator = TestHelpers::test_option_tag<
      domain::OptionTags::DomainCreator<3>,
      TestHelpers::domain::BoundaryConditions::
          MetavariablesWithoutBoundaryConditions<3, domain::creators::Sphere>>(
      "Sphere:\n"
      "  InnerRadius: 1\n"
      "  OuterRadius: 3\n"
      "  InnerCubeSphericity: 0.0\n"
      "  InitialRefinement: 2\n"
      "  InitialGridPoints: [3, 3, 4]\n"
      "  UseEquiangularMap: false\n"
      "  RadialPartitioning: []\n"
      "  RadialDistribution: [Linear]\n"
      "  TimeDependence:\n"
      "    UniformTranslation:\n"
      "      InitialTime: 1.0\n"
      "      Velocity: [2.3, -0.3, 0.5]\n");
  const auto* sphere =
      dynamic_cast<const creators::Sphere*>(domain_creator.get());

  const double inner_radius = 1.0;
  const double outer_radius = 3.0;
  const std::vector<double> radial_partitioning = {};
  const std::array<double, 3> velocity{{2.3, -0.3, 0.5}};
  const std::vector<double> times{1., 10.};

  test_sphere_construction(*sphere, inner_radius, outer_radius,
                           radial_partitioning, false, times, velocity);
}
}  // namespace

// [[TimeOut, 15]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.Sphere", "[Domain][Unit]") {
  domain::creators::time_dependence::register_derived_with_charm();
  test_parse_errors();
  test_sphere_boundaries();
  test_sphere_factory_time_dependent();
}
}  // namespace domain
