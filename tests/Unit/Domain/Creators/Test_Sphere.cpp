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
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace domain {
namespace {
using Translation3D = CoordinateMaps::TimeDependent::Translation<3>;
using Interior =
    std::variant<creators::Sphere::Excision, creators::Sphere::InnerCube>;

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_boundary_condition(const bool outer) {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      outer ? Direction<3>::upper_zeta() : Direction<3>::lower_zeta(), 50);
}

Interior copy_interior(const Interior& interior,
                       const bool with_boundary_conditions) {
  if (std::holds_alternative<creators::Sphere::InnerCube>(interior)) {
    return std::get<creators::Sphere::InnerCube>(interior);
  } else {
    return creators::Sphere::Excision{
        with_boundary_conditions ? create_boundary_condition(false) : nullptr};
  }
}

std::string stringize(const bool t) { return t ? "true" : "false"; }

template <typename T>
std::string stringize(const std::vector<T>& t) {
  std::string result = "[";
  bool first = true;
  for (const auto& item : t) {
    if (not first) {
      result += ", ";
    }
    result += get_output(item);
    first = false;
  }
  result += "]";
  return result;
}

std::string option_string(
    const double inner_radius, const double outer_radius,
    const Interior& interior, const size_t initial_refinement,
    const std::array<size_t, 3> initial_extents, const bool equiangular,
    const std::optional<creators::Sphere::EquatorialCompressionOptions>&
        equatorial_compression,
    const std::vector<double>& radial_partitioning,
    const std::vector<CoordinateMaps::Distribution>& radial_distribution,
    const ShellWedges which_wedges, const bool time_dependent,
    const bool hard_coded_time_dependent_maps,
    const bool with_boundary_conditions) {
  const std::string interior_option =
      [&interior, &with_boundary_conditions]() -> std::string {
    if (std::holds_alternative<creators::Sphere::Excision>(interior)) {
      if (with_boundary_conditions) {
        return "  Interior:\n"
               "    ExciseWithBoundaryCondition:\n"
               "      TestBoundaryCondition:\n"
               "        Direction: lower-zeta\n"
               "        BlockId: 50\n";
      } else {
        return "  Interior: Excise\n";
      }
    } else {
      const double sphericity =
          std::get<creators::Sphere::InnerCube>(interior).sphericity;
      return "  Interior:\n"
             "    FillWithSphericity: " +
             std::to_string(sphericity) + "\n";
    }
  }();
  const std::string equatorial_compression_option =
      equatorial_compression.has_value()
          ? "  EquatorialCompression:\n"
            "    AspectRatio: " +
                std::to_string(equatorial_compression->aspect_ratio) +
                "\n"
                "    IndexPolarAxis: " +
                std::to_string(equatorial_compression->index_polar_axis) + "\n"
          : "  EquatorialCompression: None\n";
  const std::string time_dependent_option =
      time_dependent ? (hard_coded_time_dependent_maps
                            ? "  TimeDependentMaps:\n"
                              "    InitialTime: 1.0\n"
                              "    SizeMap:\n"
                              "      InitialValues: [0.5, -0.04, 0.003]\n"
                              "    ShapeMap:\n"
                              "      LMax: 10\n"
                              "      InitialValues: Spherical\n"
                            : "  TimeDependentMaps:\n"
                              "    UniformTranslation:\n"
                              "      InitialTime: 1.0\n"
                              "      Velocity: [2.3, -0.3, 0.5]\n")
                     : "  TimeDependentMaps: None\n";
  const std::string outer_bc_option = with_boundary_conditions
                                          ? "  OuterBoundaryCondition:\n"
                                            "    TestBoundaryCondition:\n"
                                            "      Direction: upper-zeta\n"
                                            "      BlockId: 50\n"
                                          : "";
  return "Sphere:\n"
         "  InnerRadius: " +
         std::to_string(inner_radius) +
         "\n"
         "  OuterRadius: " +
         std::to_string(outer_radius) + "\n" + interior_option +
         "  InitialRefinement: " + std::to_string(initial_refinement) +
         "\n"
         "  InitialGridPoints: [" +
         std::to_string(initial_extents[0]) + ", " +
         std::to_string(initial_extents[1]) + ", " +
         std::to_string(initial_extents[2]) +
         "]\n"
         "  UseEquiangularMap: " +
         stringize(equiangular) +
         "\n"
         "  EquatorialCompression: None\n"
         "  WhichWedges: " +
         get_output(which_wedges) +
         "\n"
         "  RadialPartitioning: " +
         stringize(radial_partitioning) +
         "\n"
         "  RadialDistribution: " +
         (radial_distribution.size() == 1 ? get_output(radial_distribution[0])
                                          : stringize(radial_distribution)) +
         "\n" + time_dependent_option + outer_bc_option;
}

// Calculate block logical coordinates of points residing on corners of the
// inner cube or on radial block faces of wedges. The radial direction in 3D
// wedges is the positive zeta direction. These coordinates will be used to
// ensure the points lie on concentric spheres defined by either the inner
// sphere, outer sphere, or radial partition parameters.
tnsr::I<double, 3, Frame::BlockLogical> logical_coords(
    const gsl::not_null<std::mt19937*> generator, const bool is_inner_cube,
    const bool abuts_inner_cube) {
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

  if (is_inner_cube) {
    // inner cube only uses integer corners
    xi_logical_coord = rand_int_xi;
    eta_logical_coord = rand_int_eta;

  } else if (abuts_inner_cube) {
    // next to inner cube,
    // corners only on lower face b/c of square inner cube neighbor
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
    const double outer_radius, const bool fill_interior,
    const std::vector<double> radial_partitioning = {},
    const ShellWedges which_wedges = ShellWedges::All,
    const bool expect_boundary_conditions = true,
    const std::vector<double>& times = {0.},
    const std::array<double, 3>& velocity = {{0., 0., 0.}},
    const std::array<double, 3>& size_values = {{0., 0., 0.}}) {
  // check consistency of domain
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      sphere, expect_boundary_conditions, false, times);

  const auto& blocks = domain.blocks();
  const auto block_names = sphere.block_names();
  const size_t num_blocks = blocks.size();
  const size_t num_blocks_per_shell =
      which_wedges == ShellWedges::All
          ? 6
          : which_wedges == ShellWedges::FourOnEquator ? 4 : 1;
  CAPTURE(num_blocks);
  CAPTURE(num_blocks_per_shell);
  const auto all_boundary_conditions = sphere.external_boundary_conditions();
  const auto functions_of_time = sphere.functions_of_time();

  // construct vector of inner radius, outer radius, and refinements levels
  // where inertial block corners have to be located
  std::vector<double> expected_corner_radii = radial_partitioning;
  expected_corner_radii.insert(expected_corner_radii.begin(), inner_radius);
  expected_corner_radii.emplace_back(outer_radius);

  // Check total number of external boundaries
  const size_t num_shells = radial_partitioning.size() + 1;
  const size_t num_external_boundaries =
      alg::accumulate(blocks, 0_st, [](const size_t count, const auto& block) {
        return count + block.external_boundaries().size();
      });
  if (which_wedges == ShellWedges::All) {
    CHECK(num_external_boundaries == (fill_interior ? 6 : 12));
  } else if (which_wedges == ShellWedges::FourOnEquator) {
    CHECK(num_external_boundaries ==
          ((fill_interior ? 2 : 4) + 4 * (1 + num_shells * 2)));
  } else if (which_wedges == ShellWedges::OneAlongMinusX) {
    CHECK(num_external_boundaries ==
          ((fill_interior ? 5 : 1) + 1 + num_shells * 4));
  }

  MAKE_GENERATOR(generator);

  // verify if adjacent to inner cube
  const auto abuts_inner_cube =
      [&num_blocks](const auto& direction_and_neighbor) {
        return direction_and_neighbor.second.id() == num_blocks - 1;
      };

  for (size_t block_id = 0; block_id < num_blocks; ++block_id) {
    CAPTURE(block_id);
    const auto& block = blocks[block_id];
    const auto& boundary_conditions = all_boundary_conditions[block_id];
    const bool is_inner_cube = fill_interior and block_id == num_blocks - 1;

    {
      INFO("Block boundaries are spherical");
      // This section tests if the logical coordinates of corners from all
      // blocks (and points on upper wedge faces) lie on spherical shells
      // specified by inner radius, radial partitions, or outer radius
      const auto coords_on_spherical_partition = logical_coords(
          make_not_null(&generator), is_inner_cube,
          fill_interior and alg::any_of(block.neighbors(), abuts_inner_cube));
      for (const double current_time : times) {
        CAPTURE(current_time);
        const double corner_distance_from_origin =
            [&block, &coords_on_spherical_partition, &current_time, &block_id,
             &num_blocks_per_shell, &inner_radius, &outer_radius,
             &radial_partitioning, &size_values, &functions_of_time,
             &velocity]() -> double {
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
            const double target_radius =
                get(magnitude(inertial_location_time_dep));
            const double delta_t = current_time - 1.0;

            // Only when using hard coded time dependent maps do we have
            // distorted maps in the inner shell.
            if (block.has_distorted_frame()) {
              CHECK(block_id < num_blocks_per_shell);
              // This is the calculation of the inverse of the
              // SphericalCompression map since the shape map is the identity
              // currently
              const double min_radius = inner_radius;
              const double max_radius = radial_partitioning.size() > 0
                                            ? radial_partitioning[0]
                                            : outer_radius;
              const double func =
                  gsl::at(size_values, 0) + gsl::at(size_values, 1) * delta_t +
                  0.5 * gsl::at(size_values, 2) * square(delta_t);
              const double func_Y00 = func / sqrt(4.0 * M_PI);
              const double max_minus_min = max_radius - min_radius;
              const double scale =
                  (max_minus_min + func_Y00 * max_radius / target_radius) /
                  (max_minus_min + func_Y00);

              return target_radius * scale;
            } else if (block.moving_mesh_grid_to_inertial_map().is_identity()) {
              // This happens in our test if we have hard coded time dependent
              // maps, but we aren't in the first shell. Then it's just the
              // identity map
              return target_radius;
            } else {
              // This means we are using a translation time dependence.
              // origin in inertial frame (need to shift inertial
              // coord by velocity * (final_time - initial_time))
              return sqrt(square(get<0>(inertial_location_time_dep) -
                                 velocity[0] * (delta_t)) +
                          square(get<1>(inertial_location_time_dep) -
                                 velocity[1] * (delta_t)) +
                          square(get<2>(inertial_location_time_dep) -
                                 velocity[2] * (delta_t)));
            }  // end block.has_distorted_frame if/else-if/else
          }    // end time-dependent if/else
        }();   // end lambda
        CAPTURE(corner_distance_from_origin);
        CAPTURE(expected_corner_radii);
        const auto match_demarcation =
            [&corner_distance_from_origin](const double radius) {
              return corner_distance_from_origin == approx(radius);
            };
        CHECK(alg::any_of(expected_corner_radii, match_demarcation));
      }
    }

    if (which_wedges == ShellWedges::All) {
      INFO("External boundaries");
      const auto& external_boundaries = block.external_boundaries();
      CAPTURE(external_boundaries);
      if (is_inner_cube) {
        // Inner cube cannot have external boundaries
        CHECK(external_boundaries.empty());
      } else {
        // Wedges can have 0, 1, or 2 external boundaries
        std::unordered_set<size_t> allowed_num_external_boundaries{};
        if (fill_interior) {
          allowed_num_external_boundaries.insert(1);
        } else {
          allowed_num_external_boundaries.insert(2);
        }
        if (not radial_partitioning.empty()) {
          allowed_num_external_boundaries.insert(0);
          allowed_num_external_boundaries.insert(1);
        }
        CHECK(allowed_num_external_boundaries.count(
                  external_boundaries.size()) == 1);
      }
      // All external boundaries must be radial
      for (const Direction<3>& direction : external_boundaries) {
        CAPTURE(direction);
        if (fill_interior) {
          // Stronger condition for filled sphere: all external boundaries
          // must be upper zeta
          CHECK(direction == Direction<3>::upper_zeta());
        } else {
          CHECK(direction.axis() == Direction<3>::Axis::Zeta);
        }
      }
      // All angular neighbors must have the same external boundaries
      if (not is_inner_cube) {
        for (const auto& [direction, neighbor_id] : block.neighbors()) {
          CAPTURE(direction);
          if (direction.axis() != Direction<3>::Axis::Zeta) {
            CHECK(blocks[neighbor_id.id()].external_boundaries() ==
                  external_boundaries);
          }
        }
      }
    }

    if (expect_boundary_conditions) {
      INFO("Boundary conditions");
      for (const auto& direction : block.external_boundaries()) {
        CAPTURE(direction);
        const auto& boundary_condition =
            dynamic_cast<const TestHelpers::domain::BoundaryConditions::
                             TestBoundaryCondition<3>&>(
                *boundary_conditions.at(direction));
        CHECK(boundary_condition.direction() == direction);
      }
    }
  }  // block loop
}  // test_sphere_construction()

// ensure CHECK_THROWS_WITH calls are properly captured
void test_parse_errors() {
  INFO("Sphere check throws");
  const double inner_radius = 1.0;
  const double outer_radius = 2.0;
  const creators::Sphere::InnerCube inner_cube{0.0};
  const size_t refinement = 2;
  const std::array<size_t, 3> initial_extents{{4, 5, 6}};
  const bool use_equiangular_map = true;
  const std::optional<creators::Sphere::EquatorialCompressionOptions>
      equatorial_compression = std::nullopt;
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
  const ShellWedges which_wedges = ShellWedges::All;

  CHECK_THROWS_WITH(
      creators::Sphere(inner_radius, 0.5 * inner_radius, inner_cube, refinement,
                       initial_extents, use_equiangular_map,
                       equatorial_compression, radial_partitioning,
                       radial_distribution, which_wedges, std::nullopt, nullptr,
                       Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Inner radius must be smaller than outer radius"));

  CHECK_THROWS_WITH(
      creators::Sphere(inner_radius, outer_radius, inner_cube, refinement,
                       initial_extents, use_equiangular_map,
                       equatorial_compression, radial_partitioning_unordered,
                       radial_distribution, which_wedges, std::nullopt, nullptr,
                       Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Specify radial partitioning in ascending order."));

  CHECK_THROWS_WITH(
      creators::Sphere(inner_radius, outer_radius, inner_cube, refinement,
                       initial_extents, use_equiangular_map,
                       equatorial_compression, radial_partitioning_low,
                       radial_distribution, which_wedges, std::nullopt, nullptr,
                       Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "First radial partition must be larger than inner"));
  CHECK_THROWS_WITH(
      creators::Sphere(inner_radius, outer_radius, inner_cube, refinement,
                       initial_extents, use_equiangular_map,
                       equatorial_compression, radial_partitioning_high,
                       radial_distribution, which_wedges, std::nullopt, nullptr,
                       Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Last radial partition must be smaller than outer"));
  CHECK_THROWS_WITH(
      creators::Sphere(inner_radius, outer_radius, inner_cube, refinement,
                       initial_extents, use_equiangular_map,
                       equatorial_compression, radial_partitioning,
                       radial_distribution_too_many, which_wedges, std::nullopt,
                       nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Specify a 'RadialDistribution' for every spherical shell. You"));
  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, outer_radius, inner_cube, refinement, initial_extents,
          use_equiangular_map, equatorial_compression, radial_partitioning,
          radial_distribution_inner_log, which_wedges, std::nullopt, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "The 'RadialDistribution' must be 'Linear' for the"));

  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, outer_radius, inner_cube, refinement, initial_extents,
          use_equiangular_map, equatorial_compression, radial_partitioning,
          radial_distribution, which_wedges, std::nullopt,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Cannot have periodic boundary conditions with a Sphere"));
  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, outer_radius, inner_cube, refinement, initial_extents,
          use_equiangular_map, equatorial_compression, radial_partitioning,
          radial_distribution, which_wedges, std::nullopt,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "None boundary condition is not supported. If you would like "
          "an outflow-type boundary condition, you must use that."));
  CHECK_THROWS_WITH(
      creators::Sphere(inner_radius, outer_radius, inner_cube, refinement,
                       initial_extents, use_equiangular_map,
                       equatorial_compression, radial_partitioning,
                       radial_distribution, which_wedges,
                       domain::creators::sphere::TimeDependentMapOptions{
                           1.0, std::nullopt, 5, std::nullopt},
                       nullptr),
      Catch::Matchers::Contains(
          "Currently cannot use hard-coded time dependent maps with an inner "
          "cube. Use a TimeDependence instead."));
}

void test_sphere() {
  MAKE_GENERATOR(gen);

  const double inner_radius = 1.0;
  const double outer_radius = 2.0;
  const size_t initial_refinement = 3;
  const std::array<size_t, 3> initial_extents{{4, 5, 6}};

  const std::array<
      std::variant<creators::Sphere::Excision, creators::Sphere::InnerCube>, 3>
      interiors{{creators::Sphere::InnerCube{0.0},
                 creators::Sphere::InnerCube{0.7},
                 creators::Sphere::Excision{}}};
  const std::array<
      std::optional<creators::Sphere::EquatorialCompressionOptions>, 2>
      equatorial_compressions{
          {std::nullopt,
           creators::Sphere::EquatorialCompressionOptions{0.5, 2}}};
  const double outer_minus_inner = outer_radius - inner_radius;
  const std::array<std::vector<double>, 3> radial_partitioning{
      {{},
       {0.5 * (inner_radius + outer_radius)},
       {inner_radius + 0.3 * outer_minus_inner,
        inner_radius + 0.6 * outer_minus_inner}}};
  const std::array<std::vector<domain::CoordinateMaps::Distribution>, 3>
      radial_distribution{{{domain::CoordinateMaps::Distribution::Linear},
                           {domain::CoordinateMaps::Distribution::Linear,
                            domain::CoordinateMaps::Distribution::Logarithmic},
                           {domain::CoordinateMaps::Distribution::Linear}}};

  const std::array<double, 3> velocity{{2.3, -0.3, 0.5}};
  const std::array<double, 3> size_values{0.5, -0.04, 0.003};
  const size_t l_max = 10;
  const std::vector<double> times{1., 10.};

  for (auto [interior_index, equiangular, equatorial_compression, array_index,
             which_wedges, time_dependent, use_hard_coded_time_dep_options,
             with_boundary_conditions] :
       random_sample<5>(
           cartesian_product(
               make_array(0_st, 1_st, 2_st), make_array(false, true),
               equatorial_compressions, make_array(0_st, 1_st, 2_st),
               make_array(ShellWedges::All, ShellWedges::FourOnEquator,
                          ShellWedges::OneAlongMinusX),
               make_array(true, false), make_array(true, false),
               make_array(true, false)),
           make_not_null(&gen))) {
    const auto& interior = interiors[interior_index];
    const bool fill_interior =
        std::holds_alternative<creators::Sphere::InnerCube>(interior);
    CAPTURE(fill_interior);
    CAPTURE(equiangular);
    CAPTURE(radial_partitioning[array_index]);
    CAPTURE(radial_distribution[array_index]);
    CAPTURE(which_wedges);
    CAPTURE(time_dependent);
    CAPTURE(with_boundary_conditions);
    // If we aren't time dependent, just set the hard coded option to false to
    // avoid ambiguity. If we are time dependent but we're filling the interior,
    // then we can't use hard coded options
    if ((not time_dependent) or fill_interior) {
      use_hard_coded_time_dep_options = false;
    }
    CAPTURE(use_hard_coded_time_dep_options);

    if (which_wedges != ShellWedges::All and with_boundary_conditions) {
      continue;
    }

    creators::Sphere::RadialDistribution::type radial_distribution_variant;
    if (radial_distribution[array_index].size() == 1) {
      radial_distribution_variant = radial_distribution[array_index][0];
    } else {
      radial_distribution_variant = radial_distribution[array_index];
    }

    std::optional<creators::Sphere::TimeDepOptionType> time_dependent_options{};

    if (time_dependent) {
      if (use_hard_coded_time_dep_options) {
        time_dependent_options = creators::sphere::TimeDependentMapOptions(
            1.0, size_values, l_max, std::nullopt);
      } else {
        time_dependent_options = std::make_unique<
            domain::creators::time_dependence::UniformTranslation<3, 0>>(
            1.0, velocity);
      }
    }

    const creators::Sphere sphere{
        inner_radius,
        outer_radius,
        copy_interior(interior, with_boundary_conditions),
        initial_refinement,
        initial_extents,
        equiangular,
        equatorial_compression,
        radial_partitioning[array_index],
        radial_distribution_variant,
        which_wedges,
        std::move(time_dependent_options),
        with_boundary_conditions ? create_boundary_condition(true) : nullptr};
    test_sphere_construction(
        sphere, inner_radius, outer_radius, fill_interior,
        radial_partitioning[array_index], which_wedges,
        with_boundary_conditions,
        time_dependent ? times : std::vector<double>{0.},
        time_dependent ? velocity : std::array<double, 3>{{0., 0., 0.}},
        time_dependent ? size_values : std::array<double, 3>{{0., 0., 0.}});
    TestHelpers::domain::creators::test_creation(
        option_string(inner_radius, outer_radius, interior, initial_refinement,
                      initial_extents, equiangular, equatorial_compression,
                      radial_partitioning[array_index],
                      radial_distribution[array_index], which_wedges,
                      time_dependent, use_hard_coded_time_dep_options,
                      with_boundary_conditions),
        sphere, with_boundary_conditions);
  }
}

}  // namespace

// [[TimeOut, 15]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.Sphere", "[Domain][Unit]") {
  domain::creators::time_dependence::register_derived_with_charm();
  test_parse_errors();
  test_sphere();
}
}  // namespace domain
