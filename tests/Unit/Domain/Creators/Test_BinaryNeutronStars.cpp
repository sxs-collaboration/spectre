// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/BinaryNeutronStars.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/Creators/TimeDependentOptions/ExpansionMap.hpp"
#include "Domain/Creators/TimeDependentOptions/RotationMap.hpp"
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/Creators/TimeDependentOptions/TranslationMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/NeighborIsConforming.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/QuadratureWeights.hpp"
#include "Options/Context.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace {
using BinaryNeutronStars = domain::creators::BinaryNeutronStars;
using TimeDependentMapOptions =
    domain::creators::bco::TimeDependentMapOptions<true>;

// Constructor arguments except the time-dependent maps and boundary condition.
// The defaults are valid for both values of `bulge`.
struct Parameters {
  std::array<double, 4> x_axis_points{-6.9, -4.0, 3.1, 5.8};
  double wedge_inner_radius = 2.9;
  double wedge_outer_radius = 7.0;
  double cylinder_outer_radius = 15.0;
  double outer_radius = 20.0;
  std::array<size_t, 4> cube_grid_points{3, 4, 5, 3};
  std::array<size_t, 4> cube_x_refinement{1, 0, 1, 2};
  std::array<size_t, 4> cube_yz_refinement{0, 1, 0, 1};
  size_t wedge_prism_radial_grid_points = 3;
  size_t wedge_prism_radial_refinement = 1;
  size_t cylinder_radial_grid_points = 3;
  size_t cylinder_radial_refinement = 1;
  size_t b2_angular_grid_points = 7;
  std::array<size_t, 2> hollow_cylinder_angular_grid_points{7, 3};
  BinaryNeutronStars::SphericalShellsInitialRadialRefinement::type
      spherical_shells_radial_refinement = 2_st;
  size_t spherical_shells_radial_grid_points = 3;
  size_t spherical_harmonic_l = 6;
  std::vector<double> spherical_shells_radial_partitioning{};
  BinaryNeutronStars::SphericalShellsRadialDistribution::type
      spherical_shells_radial_distribution =
          domain::CoordinateMaps::Distribution::Linear;
  bool bulge = false;

  double left_end() const { return x_axis_points[0]; }
  double center_B() const { return x_axis_points[1]; }
  double center_A() const { return x_axis_points[2]; }
  double right_end() const { return x_axis_points[3]; }
};

// Equal mass (|CenterA| == |CenterB|) with two spherical shells
Parameters equal_mass_parameters(const bool bulge) {
  Parameters params{};
  params.x_axis_points = {-6.0, -3.5, 3.5, 6.0};
  params.wedge_inner_radius = 2.5;
  params.spherical_shells_radial_partitioning = {17.0};
  params.spherical_shells_radial_refinement = std::vector<size_t>{1, 2};
  params.spherical_shells_radial_distribution =
      std::vector<domain::CoordinateMaps::Distribution>{
          domain::CoordinateMaps::Distribution::Linear,
          domain::CoordinateMaps::Distribution::Logarithmic};
  params.bulge = bulge;
  return params;
}

BinaryNeutronStars make_creator(
    const Parameters& params,
    std::optional<TimeDependentMapOptions> time_dependent_options =
        std::nullopt,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> outer_bc =
        nullptr,
    const Options::Context& context = {}) {
  return BinaryNeutronStars{params.x_axis_points,
                            params.wedge_inner_radius,
                            params.wedge_outer_radius,
                            params.cylinder_outer_radius,
                            params.outer_radius,
                            params.cube_grid_points,
                            params.cube_x_refinement,
                            params.cube_yz_refinement,
                            params.wedge_prism_radial_grid_points,
                            params.wedge_prism_radial_refinement,
                            params.cylinder_radial_grid_points,
                            params.cylinder_radial_refinement,
                            params.b2_angular_grid_points,
                            params.hollow_cylinder_angular_grid_points,
                            params.spherical_shells_radial_refinement,
                            params.spherical_shells_radial_grid_points,
                            params.spherical_harmonic_l,
                            params.spherical_shells_radial_partitioning,
                            params.spherical_shells_radial_distribution,
                            params.bulge,
                            std::move(time_dependent_options),
                            std::move(outer_bc),
                            context};
}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_outer_boundary_condition() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::upper_xi(), 50);
}

// With Bulge=true, the radii of the wedges' outer spheres around B and A
std::array<double, 2> bulge_radii(const Parameters& params) {
  const double radius_B = params.wedge_outer_radius;
  const double radius_A = sqrt(square(radius_B) - square(params.center_B()) +
                               square(params.center_A()));
  return {radius_B, radius_A};
}

// The x-coordinates bounding the four cubed-cylinder groups
std::array<double, 5> group_boundaries(const Parameters& params) {
  return {params.left_end(), params.center_B(), 0.0, params.center_A(),
          params.right_end()};
}

std::vector<std::string> expected_block_names() {
  return {// BLeft cubed-cylinder (blocks 0-4)
          "BLeftCubedCylinderCenter", "BLeftCubedCylinderFront",
          "BLeftCubedCylinderTop", "BLeftCubedCylinderBack",
          "BLeftCubedCylinderBottom",
          // BRight cubed-cylinder (blocks 5-9)
          "BRightCubedCylinderCenter", "BRightCubedCylinderFront",
          "BRightCubedCylinderTop", "BRightCubedCylinderBack",
          "BRightCubedCylinderBottom",
          // ALeft cubed-cylinder (blocks 10-14)
          "ALeftCubedCylinderCenter", "ALeftCubedCylinderFront",
          "ALeftCubedCylinderTop", "ALeftCubedCylinderBack",
          "ALeftCubedCylinderBottom",
          // ARight cubed-cylinder (blocks 15-19)
          "ARightCubedCylinderCenter", "ARightCubedCylinderFront",
          "ARightCubedCylinderTop", "ARightCubedCylinderBack",
          "ARightCubedCylinderBottom",
          // Cylinders (blocks 20-25)
          "BFilledCylinder", "BLeftHollowCylinder", "BRightHollowCylinder",
          "ALeftHollowCylinder", "ARightHollowCylinder", "AFilledCylinder",
          // Spherical shell (block 26)
          "SphericalShell0"};
}

void test_block_names_and_groups() {
  INFO("Test block names and groups");
  // Same for both values of bulge
  const auto creator = make_creator(Parameters{});
  const auto names = creator.block_names();
  CHECK(names == expected_block_names());
  REQUIRE(names.size() == 27);

  const auto groups = creator.block_groups();
  // CubedCylinders: 20 blocks (4 groups of 5)
  REQUIRE(groups.count("CubedCylinders") == 1);
  CHECK(groups.at("CubedCylinders").size() == 20);
  CHECK(groups.at("CubedCylinders").count("BLeftCubedCylinderCenter") == 1);
  CHECK(groups.at("CubedCylinders").count("ARightCubedCylinderBottom") == 1);
  // Wedges: the 16 non-central cubed-cylinder blocks
  REQUIRE(groups.count("Wedges") == 1);
  CHECK(groups.at("Wedges").size() == 16);
  CHECK(not groups.at("Wedges").contains("BLeftCubedCylinderCenter"));
  CHECK(groups.at("Wedges").count("ARightCubedCylinderBottom") == 1);
  for (const auto& wedge_name : groups.at("Wedges")) {
    CHECK(groups.at("CubedCylinders").count(wedge_name) == 1);
  }
  // Cylinders: 6 blocks (2 filled + 4 hollow)
  REQUIRE(groups.count("Cylinders") == 1);
  CHECK(groups.at("Cylinders") ==
        std::unordered_set<std::string>{
            "BFilledCylinder", "BLeftHollowCylinder", "BRightHollowCylinder",
            "ALeftHollowCylinder", "ARightHollowCylinder", "AFilledCylinder"});
  // SphericalShells: 1 block
  REQUIRE(groups.count("SphericalShells") == 1);
  CHECK(groups.at("SphericalShells") ==
        std::unordered_set<std::string>{"SphericalShell0"});

  // The object centers and the midpoint between them
  const auto anchors = creator.grid_anchors();
  CHECK(anchors.size() == 3);
  CHECK(anchors.at("Center") == tnsr::I<double, 3, Frame::Grid>{
                                    std::array{0.5 * (3.1 - 4.0), 0.0, 0.0}});
  CHECK(anchors.at("CenterA") ==
        tnsr::I<double, 3, Frame::Grid>{std::array{3.1, 0.0, 0.0}});
  CHECK(anchors.at("CenterB") ==
        tnsr::I<double, 3, Frame::Grid>{std::array{-4.0, 0.0, 0.0}});
}

void test_domain_structure(const bool bulge) {
  INFO("Test domain structure (bulge=" << bulge << ")");
  Parameters params{};
  params.bulge = bulge;
  const auto domain = make_creator(params).create_domain();
  const auto& blocks = domain.blocks();
  REQUIRE(blocks.size() == 27);

  CHECK(domain.excision_spheres().empty());

  for (size_t i = 0; i < blocks.size(); ++i) {
    CAPTURE(i);
    CAPTURE(blocks[i].name());
    if (i < 20) {
      CHECK(blocks[i].topologies() == domain::topologies::hypercube<3>);
    } else if (i == 20 or i == 25) {
      CHECK(blocks[i].topologies() == domain::topologies::full_cylinder);
    } else if (i < 25) {
      CHECK(blocks[i].topologies() == domain::topologies::cylindrical_shell);
    } else {
      CHECK(blocks[i].topologies() == domain::topologies::spherical_shell);
    }
    // Only the outer face of the spherical shell is external
    const auto& ext = blocks[i].external_boundaries();
    if (i == 26) {
      CHECK(ext.size() == 1);
      CHECK(ext.count(Direction<3>::upper_xi()) == 1);
    } else {
      CHECK(ext.empty());
    }
  }
}

void test_initial_extents_and_refinement() {
  INFO("Test initial extents and refinement");
  using Array = std::array<size_t, 3>;
  for (const bool bulge : {false, true}) {
    CAPTURE(bulge);
    Parameters params{};
    params.bulge = bulge;
    const auto creator = make_creator(params);
    const auto extents = creator.initial_extents();
    const auto refinement = creator.initial_refinement_levels();
    REQUIRE(extents.size() == 27);
    REQUIRE(refinement.size() == 27);
    // Central cubes
    CHECK(extents[0] == Array{3, 3, 3});
    CHECK(refinement[0] == Array{1, 0, 0});
    CHECK(extents[5] == Array{4, 4, 4});
    CHECK(refinement[5] == Array{0, 1, 1});
    CHECK(extents[10] == Array{5, 5, 5});
    CHECK(refinement[10] == Array{1, 0, 0});
    CHECK(extents[15] == Array{3, 3, 3});
    CHECK(refinement[15] == Array{2, 1, 1});
    // Wedges: [radial, angular, axial] for Bulge=false and
    // [axial, angular, radial] for Bulge=true
    for (size_t j = 1; j < 5; ++j) {
      CAPTURE(j);
      CHECK(extents[10 + j] == (bulge ? Array{5, 5, 3} : Array{3, 5, 5}));
      CHECK(refinement[15 + j] == (bulge ? Array{2, 1, 1} : Array{1, 1, 2}));
    }
    // Filled cylinders: [B2 radial, B2 angular, axial]
    for (const size_t i : {20_st, 25_st}) {
      CAPTURE(i);
      CHECK(extents[i] == Array{3, 7, 3});
      CHECK(refinement[i] == Array{0, 0, 1});
    }
    // Hollow cylinders: [radial, angular, axial]
    for (const size_t i : {21_st, 22_st, 23_st, 24_st}) {
      CAPTURE(i);
      CHECK(extents[i] == Array{3, 7, 3});
      CHECK(refinement[i] == Array{1, 0, 0});
    }
    // Spherical shell: [radial, L + 1, 2L + 1]
    CHECK(extents[26] == Array{3, 7, 13});
    CHECK(refinement[26] == Array{2, 0, 0});
  }
  {
    INFO("Per-shell radial refinement");
    const auto refinement =
        make_creator(equal_mass_parameters(false)).initial_refinement_levels();
    REQUIRE(refinement.size() == 28);
    CHECK(refinement[26] == Array{1, 0, 0});
    CHECK(refinement[27] == Array{2, 0, 0});
  }
}

void test_domain_creator_checks() {
  INFO("Test domain creator consistency checks");
  for (const bool bulge : {false, true}) {
    CAPTURE(bulge);
    Parameters params{};
    params.bulge = bulge;
    {
      INFO("No BC");
      const auto creator = make_creator(params);
      TestHelpers::domain::creators::test_domain_creator(creator, false);
      test_non_conforming_interface_logical_coords(
          creator.create_domain().blocks());
    }
    {
      INFO("With BC");
      const auto creator =
          make_creator(params, std::nullopt, create_outer_boundary_condition());
      TestHelpers::domain::creators::test_domain_creator(creator, true);
    }
    {
      INFO("Equal mass, multiple shells");
      const auto creator =
          make_creator(equal_mass_parameters(bulge), std::nullopt,
                       create_outer_boundary_condition());
      TestHelpers::domain::creators::test_domain_creator(creator, true);
      test_non_conforming_interface_logical_coords(
          creator.create_domain().blocks());
    }
  }
}

// Quadrature points and weights for one logical direction of a block:
// Gauss-Legendre for bounded directions (its points avoid singularities at the
// endpoints, such as the poles) and uniform for periodic ones.
std::pair<DataVector, DataVector> logical_quadrature(
    const domain::Topology topology, const size_t number_of_points) {
  const auto gauss_legendre = [number_of_points](const double lower,
                                                 const double upper) {
    const Mesh<1> mesh{number_of_points, Spectral::Basis::Legendre,
                       Spectral::Quadrature::Gauss};
    return std::pair{
        DataVector{0.5 * (upper + lower) +
                   0.5 * (upper - lower) * Spectral::collocation_points(mesh)},
        DataVector{0.5 * (upper - lower) * Spectral::quadrature_weights(mesh)}};
  };
  switch (topology) {
    case domain::Topology::I1:
    case domain::Topology::B2Radial:
      return gauss_legendre(-1.0, 1.0);
    case domain::Topology::S2Colatitude: {
      // The Jacobian of SphericalToCartesianPfaffian omits the sin(theta) of
      // the volume element
      auto [points, weights] = gauss_legendre(0.0, std::numbers::pi);
      weights *= sin(points);
      return std::pair{std::move(points), std::move(weights)};
    }
    case domain::Topology::S1:
    case domain::Topology::B2Angular:
    case domain::Topology::S2Longitude: {
      const double spacing =
          2.0 * std::numbers::pi / static_cast<double>(number_of_points);
      DataVector points{number_of_points};
      for (size_t i = 0; i < number_of_points; ++i) {
        points[i] = spacing * static_cast<double>(i);
      }
      return std::pair{std::move(points),
                       DataVector{number_of_points, spacing}};
    }
    default:
      ERROR("Unexpected topology " << topology);
  }
}

// The volume of a block, from integrating the Jacobian determinant of its
// stationary map over its logical coordinates
double block_volume(const Block<3>& block) {
  constexpr size_t number_of_points = 16;
  std::array<std::pair<DataVector, DataVector>, 3> quadratures{};
  for (size_t d = 0; d < 3; ++d) {
    gsl::at(quadratures, d) =
        logical_quadrature(gsl::at(block.topologies(), d), number_of_points);
  }
  const size_t total_points = cube(number_of_points);
  tnsr::I<DataVector, 3, Frame::BlockLogical> logical_coords{total_points};
  DataVector weights{total_points};
  for (size_t k = 0; k < number_of_points; ++k) {
    for (size_t j = 0; j < number_of_points; ++j) {
      for (size_t i = 0; i < number_of_points; ++i) {
        const size_t index = i + number_of_points * (j + number_of_points * k);
        get<0>(logical_coords)[index] = quadratures[0].first[i];
        get<1>(logical_coords)[index] = quadratures[1].first[j];
        get<2>(logical_coords)[index] = quadratures[2].first[k];
        weights[index] = quadratures[0].second[i] * quadratures[1].second[j] *
                         quadratures[2].second[k];
      }
    }
  }
  const auto jacobian = block.stationary_map().jacobian(logical_coords);
  const DataVector det_jacobian = get(determinant(jacobian));
  {
    // The shared helpers only check corners of hypercube blocks
    INFO("Jacobian determinant must be positive throughout the block");
    CAPTURE(block.name());
    CHECK(min(det_jacobian) > 0.0);
  }
  return sum(det_jacobian * weights);
}

// The blocks must fill the ball of radius OuterRadius, and each region must
// have the volume implied by the options
void test_volumes(const Parameters& params) {
  INFO("Test volumes");
  const auto domain = make_creator(params).create_domain();
  const auto& blocks = domain.blocks();
  std::vector<double> volumes(blocks.size());
  for (size_t i = 0; i < blocks.size(); ++i) {
    volumes[i] = block_volume(blocks[i]);
  }
  Approx custom_approx = Approx::custom().epsilon(1.e-10).scale(1.0);
  const auto ball_volume = [](const double radius) {
    return 4.0 / 3.0 * std::numbers::pi * cube(radius);
  };
  const auto sum_volumes = [&volumes](const size_t first, const size_t last) {
    double result = 0.0;
    for (size_t i = first; i < last; ++i) {
      result += volumes[i];
    }
    return result;
  };

  CHECK(sum_volumes(0, blocks.size()) ==
        custom_approx(ball_volume(params.outer_radius)));
  CHECK(sum_volumes(0, 26) ==
        custom_approx(ball_volume(params.cylinder_outer_radius)));

  // Each cubed-cylinder group is bounded by a cylinder of radius
  // WedgeOuterRadius (Bulge=false) or the sphere around its object (Bulge=true)
  const auto boundaries = group_boundaries(params);
  const auto [radius_B, radius_A] = bulge_radii(params);
  for (size_t group = 0; group < 4; ++group) {
    CAPTURE(group);
    const double lower = gsl::at(boundaries, group);
    const double upper = gsl::at(boundaries, group + 1);
    double expected_volume = 0.0;
    if (params.bulge) {
      const double center = group < 2 ? params.center_B() : params.center_A();
      const double radius = group < 2 ? radius_B : radius_A;
      // pi * int_lower^upper (radius^2 - (x - center)^2) dx
      expected_volume = std::numbers::pi *
                        (square(radius) * (upper - lower) -
                         (cube(upper - center) - cube(lower - center)) / 3.0);
    } else {
      expected_volume = std::numbers::pi * square(params.wedge_outer_radius) *
                        (upper - lower);
    }
    CHECK(sum_volumes(5 * group, 5 * group + 5) ==
          custom_approx(expected_volume));
    // The central cube has half-width WedgeInnerRadius / sqrt(2)
    CHECK(volumes[5 * group] ==
          custom_approx(2.0 * square(params.wedge_inner_radius) *
                        (upper - lower)));
  }

  // Spherical shells
  double inner_radius = params.cylinder_outer_radius;
  for (size_t shell = 0; 26 + shell < blocks.size(); ++shell) {
    CAPTURE(shell);
    const double outer_radius =
        26 + shell + 1 < blocks.size()
            ? params.spherical_shells_radial_partitioning[shell]
            : params.outer_radius;
    CHECK(volumes[26 + shell] ==
          custom_approx(ball_volume(outer_radius) - ball_volume(inner_radius)));
    inner_radius = outer_radius;
  }
}

// Every point in the ball of radius OuterRadius must lie in exactly one block
template <typename Generator>
void test_points_in_exactly_one_block(const gsl::not_null<Generator*> gen,
                                      const Parameters& params) {
  INFO("Test that each point lies in exactly one block");
  const auto domain = make_creator(params).create_domain();
  const auto& blocks = domain.blocks();
  Approx custom_approx = Approx::custom().epsilon(1.e-10).scale(1.0);
  std::uniform_real_distribution<double> dist(-params.outer_radius,
                                              params.outer_radius);
  size_t number_of_points_tested = 0;
  while (number_of_points_tested < 200) {
    const tnsr::I<double, 3, Frame::Inertial> point{
        std::array{dist(*gen), dist(*gen), dist(*gen)}};
    if (get(magnitude(point)) >= params.outer_radius) {
      continue;
    }
    ++number_of_points_tested;
    CAPTURE(point);
    size_t number_of_containing_blocks = 0;
    for (const auto& block : blocks) {
      const auto logical_point =
          block_logical_coordinates_single_point(point, block);
      if (not logical_point.has_value()) {
        continue;
      }
      CAPTURE(block.name());
      CHECK_ITERABLE_CUSTOM_APPROX(block.stationary_map()(*logical_point),
                                   point, custom_approx);
      ++number_of_containing_blocks;
    }
    CHECK(number_of_containing_blocks == 1);
  }
}

// Points on a block face at the Gauss points of a 2D mesh
tnsr::I<DataVector, 3, Frame::Inertial> face_coords(
    const Block<3>& block, const Direction<3>& direction) {
  const Mesh<2> face_mesh{4, Spectral::Basis::Legendre,
                          Spectral::Quadrature::Gauss};
  const auto interface_coords =
      interface_logical_coordinates(face_mesh, direction);
  tnsr::I<DataVector, 3, Frame::BlockLogical> logical_coords{};
  for (size_t d = 0; d < 3; ++d) {
    logical_coords.get(d) = interface_coords.get(d);
  }
  return block.stationary_map()(logical_coords);
}

// The block faces must lie on the surfaces set by the options
void test_surfaces(const Parameters& params) {
  INFO("Test that block faces lie on the expected surfaces");
  const auto domain = make_creator(params).create_domain();
  const auto& blocks = domain.blocks();
  const auto boundaries = group_boundaries(params);
  const auto [radius_B, radius_A] = bulge_radii(params);
  const auto constant = [](const DataVector& like, const double value) {
    return DataVector(like.size(), value);
  };

  for (size_t group = 0; group < 4; ++group) {
    CAPTURE(group);
    // Central cube ends
    const auto& central_cube = blocks[5 * group];
    const auto lower_face = face_coords(central_cube, Direction<3>::lower_xi());
    const auto upper_face = face_coords(central_cube, Direction<3>::upper_xi());
    CHECK_ITERABLE_APPROX(
        get<0>(lower_face),
        constant(get<0>(lower_face), gsl::at(boundaries, group)));
    CHECK_ITERABLE_APPROX(
        get<0>(upper_face),
        constant(get<0>(upper_face), gsl::at(boundaries, group + 1)));
    // Outer faces of the deformed-cube wedges
    for (size_t j = 1; j < 5; ++j) {
      CAPTURE(j);
      const auto outer_face = face_coords(
          blocks[5 * group + j],
          params.bulge ? Direction<3>::upper_zeta() : Direction<3>::upper_xi());
      if (params.bulge) {
        const double center = group < 2 ? params.center_B() : params.center_A();
        const double radius = group < 2 ? radius_B : radius_A;
        CHECK_ITERABLE_APPROX(
            sqrt(square(get<0>(outer_face) - center) +
                 square(get<1>(outer_face)) + square(get<2>(outer_face))),
            constant(get<0>(outer_face), radius));
      } else {
        CHECK_ITERABLE_APPROX(
            sqrt(square(get<1>(outer_face)) + square(get<2>(outer_face))),
            constant(get<0>(outer_face), params.wedge_outer_radius));
      }
    }
  }

  // The flat faces of the filled cylinders cap the cubed cylinders
  const auto b_flat_face = face_coords(blocks[20], Direction<3>::upper_zeta());
  CHECK_ITERABLE_APPROX(get<0>(b_flat_face),
                        constant(get<0>(b_flat_face), params.left_end()));
  const auto a_flat_face = face_coords(blocks[25], Direction<3>::lower_zeta());
  CHECK_ITERABLE_APPROX(get<0>(a_flat_face),
                        constant(get<0>(a_flat_face), params.right_end()));

  // Outer faces of the cylinders
  const auto check_radius =
      [&constant](const tnsr::I<DataVector, 3, Frame::Inertial>& coords,
                  const double radius) {
        CHECK_ITERABLE_APPROX(get(magnitude(coords)),
                              constant(get<0>(coords), radius));
      };
  check_radius(face_coords(blocks[20], Direction<3>::lower_zeta()),
               params.cylinder_outer_radius);
  check_radius(face_coords(blocks[25], Direction<3>::upper_zeta()),
               params.cylinder_outer_radius);
  for (const size_t i : {21_st, 22_st, 23_st, 24_st}) {
    CAPTURE(i);
    check_radius(face_coords(blocks[i], Direction<3>::upper_xi()),
                 params.cylinder_outer_radius);
  }

  // Spherical shells
  double inner_radius = params.cylinder_outer_radius;
  for (size_t shell = 0; 26 + shell < blocks.size(); ++shell) {
    CAPTURE(shell);
    const double outer_radius =
        26 + shell + 1 < blocks.size()
            ? params.spherical_shells_radial_partitioning[shell]
            : params.outer_radius;
    check_radius(face_coords(blocks[26 + shell], Direction<3>::lower_xi()),
                 inner_radius);
    check_radius(face_coords(blocks[26 + shell], Direction<3>::upper_xi()),
                 outer_radius);
    inner_radius = outer_radius;
  }
}

bool is_angular(const domain::Topology topology) {
  return topology == domain::Topology::S1 or
         topology == domain::Topology::B2Angular or
         topology == domain::Topology::S2Longitude or
         topology == domain::Topology::S2Colatitude;
}

// Logical coordinates of 3x3 points on a face of the element of `block` with
// the given segment ids. Angular directions are never refined.
tnsr::I<DataVector, 3, Frame::BlockLogical> face_logical_coords(
    const Block<3>& block, const Direction<3>& direction,
    const std::array<SegmentId, 3>& segment_ids = {
        SegmentId{0, 0}, SegmentId{0, 0}, SegmentId{0, 0}}) {
  constexpr size_t points_per_dim = 3;
  const std::array<double, points_per_dim> interval_samples{-0.77, 0.1, 0.83};
  const std::array<double, points_per_dim> longitude_samples{0.3, 2.5, 4.4};
  const std::array<double, points_per_dim> colatitude_samples{0.4, 1.6, 2.8};
  const auto samples = [&](const size_t d) {
    std::array<double, points_per_dim> result{};
    const auto topology = gsl::at(block.topologies(), d);
    for (size_t i = 0; i < points_per_dim; ++i) {
      if (topology == domain::Topology::S2Colatitude) {
        gsl::at(result, i) = gsl::at(colatitude_samples, i);
      } else if (is_angular(topology)) {
        gsl::at(result, i) = gsl::at(longitude_samples, i);
      } else {
        const auto& segment_id = gsl::at(segment_ids, d);
        gsl::at(result, i) =
            segment_id.midpoint() + 0.5 *
                                        (segment_id.endpoint(Side::Upper) -
                                         segment_id.endpoint(Side::Lower)) *
                                        gsl::at(interval_samples, i);
      }
    }
    return result;
  };
  const size_t normal_dim = direction.dimension();
  const size_t first_dim = normal_dim == 0 ? 1 : 0;
  const size_t second_dim = normal_dim == 2 ? 1 : 2;
  const auto first_samples = samples(first_dim);
  const auto second_samples = samples(second_dim);
  tnsr::I<DataVector, 3, Frame::BlockLogical> result{square(points_per_dim)};
  for (size_t j = 0; j < points_per_dim; ++j) {
    for (size_t i = 0; i < points_per_dim; ++i) {
      const size_t index = i + points_per_dim * j;
      // All faces checked here are at +/-1 (B2 radial faces only at +1)
      result.get(normal_dim)[index] = direction.sign();
      result.get(first_dim)[index] = gsl::at(first_samples, i);
      result.get(second_dim)[index] = gsl::at(second_samples, j);
    }
  }
  return result;
}

tnsr::I<double, 3, Frame::Inertial> point(
    const tnsr::I<DataVector, 3, Frame::Inertial>& points, const size_t index) {
  return tnsr::I<double, 3, Frame::Inertial>{std::array{
      get<0>(points)[index], get<1>(points)[index], get<2>(points)[index]}};
}

// The conforming flags must agree with domain::neighbor_is_conforming, which
// create_initial_element uses for single block neighbors
void test_conforming_flags(const Parameters& params) {
  INFO("Test that the conforming flags agree with the block topologies");
  const auto domain = make_creator(params).create_domain();
  const auto& blocks = domain.blocks();
  for (const auto& block : blocks) {
    for (const auto& [direction, block_neighbors] : block.neighbors()) {
      CAPTURE(block.name());
      CAPTURE(direction);
      if (block_neighbors.size() > 1) {
        CHECK(not block_neighbors.are_conforming());
        continue;
      }
      const size_t neighbor_id = *block_neighbors.ids().begin();
      CAPTURE(blocks[neighbor_id].name());
      CHECK(domain::neighbor_is_conforming(
                block.topologies(), blocks[neighbor_id].topologies(), direction,
                block_neighbors.orientation(neighbor_id)) ==
            block_neighbors.are_conforming());
    }
  }
}

// At each conforming interface, face points must map to the neighbor's face at
// the logical coordinates given by the OrientationMap. (The shared helpers skip
// pairs of spherical shells.)
void test_conforming_interfaces(const Parameters& params) {
  INFO("Test conforming interfaces");
  const auto domain = make_creator(params).create_domain();
  const auto& blocks = domain.blocks();
  Approx custom_approx = Approx::custom().epsilon(1.e-10).scale(1.0);
  for (const auto& block : blocks) {
    for (const auto& [direction, block_neighbors] : block.neighbors()) {
      if (not block_neighbors.are_conforming()) {
        continue;
      }
      CAPTURE(block.name());
      CAPTURE(direction);
      const size_t neighbor_id = *block_neighbors.ids().begin();
      const auto& neighbor = blocks[neighbor_id];
      CAPTURE(neighbor.name());
      const auto& orientation = block_neighbors.orientation(neighbor_id);
      const auto neighbor_direction = orientation(direction).opposite();
      const auto logical_coords = face_logical_coords(block, direction);
      const auto inertial_coords = block.stationary_map()(logical_coords);
      for (size_t i = 0; i < get<0>(logical_coords).size(); ++i) {
        const auto neighbor_logical = block_logical_coordinates_single_point(
            point(inertial_coords, i), neighbor);
        REQUIRE(neighbor_logical.has_value());
        CHECK(neighbor_logical->get(neighbor_direction.dimension()) ==
              custom_approx(neighbor_direction.sign()));
        for (size_t d = 0; d < 3; ++d) {
          if (d == direction.dimension()) {
            continue;
          }
          CAPTURE(d);
          const auto mapped = orientation(Direction<3>{d, Side::Upper});
          const double expected = mapped.sign() * logical_coords.get(d)[i];
          const double actual = neighbor_logical->get(mapped.dimension());
          if (is_angular(gsl::at(block.topologies(), d))) {
            CHECK(std::remainder(actual - expected, 2.0 * std::numbers::pi) ==
                  custom_approx(0.0));
          } else {
            CHECK(actual == custom_approx(expected));
          }
        }
      }
    }
  }
}

bool element_contains(const ElementId<3>& element_id,
                      const tnsr::I<double, 3, Frame::BlockLogical>& point) {
  for (size_t d = 0; d < 3; ++d) {
    const auto& segment_id = element_id.segment_id(d);
    // Unrefined directions, including all angular ones, span the block
    if (segment_id.refinement_level() == 0) {
      continue;
    }
    if (point.get(d) < segment_id.endpoint(Side::Lower) - 1.e-10 or
        point.get(d) > segment_id.endpoint(Side::Upper) + 1.e-10) {
      return false;
    }
  }
  return true;
}

// Every point on a non-conforming face of an initial element must lie in one
// of its neighbor elements
void test_element_non_conforming_neighbors(const Parameters& params) {
  INFO("Test non-conforming element neighbors cover the element faces");
  const auto creator = make_creator(params);
  const auto domain = creator.create_domain();
  const auto& blocks = domain.blocks();
  const auto refinement = creator.initial_refinement_levels();
  size_t number_of_faces_tested = 0;
  for (const auto& element_id : initial_element_ids(refinement)) {
    const auto element =
        domain::create_initial_element(element_id, blocks, refinement);
    const auto& block = blocks[element_id.block_id()];
    for (const auto& [direction, neighbors] : element.neighbors()) {
      if (neighbors.are_conforming()) {
        continue;
      }
      ++number_of_faces_tested;
      CAPTURE(element_id);
      CAPTURE(direction);
      const auto inertial_coords = block.stationary_map()(
          face_logical_coords(block, direction, element_id.segment_ids()));
      for (size_t i = 0; i < get<0>(inertial_coords).size(); ++i) {
        const auto x = point(inertial_coords, i);
        CAPTURE(x);
        CHECK(alg::any_of(neighbors.ids(), [&](const ElementId<3>& neighbor) {
          const auto neighbor_logical = block_logical_coordinates_single_point(
              x, blocks[neighbor.block_id()]);
          return neighbor_logical.has_value() and
                 element_contains(neighbor, *neighbor_logical);
        }));
      }
    }
  }
  CHECK(number_of_faces_tested > 0);
}

void test_geometry() {
  INFO("Test geometry");
  MAKE_GENERATOR(gen);
  for (const bool bulge : {false, true}) {
    CAPTURE(bulge);
    Parameters asymmetric{};
    asymmetric.bulge = bulge;
    for (const auto& params : {asymmetric, equal_mass_parameters(bulge)}) {
      CAPTURE(params.x_axis_points[0]);
      test_volumes(params);
      test_points_in_exactly_one_block(make_not_null(&gen), params);
      test_surfaces(params);
      test_conforming_flags(params);
      test_conforming_interfaces(params);
      test_element_non_conforming_neighbors(params);
    }
  }
}

void test_option_parsing() {
  INFO("Test option parsing");
  TestHelpers::domain::creators::test_creation(
      "BinaryNeutronStars:\n"
      "  XAxisPoints: [-6.9, -4.0, 3.1, 5.8]\n"
      "  WedgeInnerRadius: 2.9\n"
      "  WedgeOuterRadius: 7.0\n"
      "  CylinderOuterRadius: 15.0\n"
      "  OuterRadius: 20.0\n"
      "  CubeInitialGridPoints: [3, 4, 5, 3]\n"
      "  CubeInitialXRefinement: [1, 0, 1, 2]\n"
      "  CubeInitialYZRefinement: [0, 1, 0, 1]\n"
      "  WedgePrismInitialRadialGridPoints: 3\n"
      "  WedgePrismInitialRadialRefinement: 1\n"
      "  CylinderInitialRadialGridPoints: 3\n"
      "  CylinderInitialRadialRefinement: 1\n"
      "  B2InitialAngularGridPoints: 7\n"
      "  HollowCylinderInitialAngularGridPoints: [7, 3]\n"
      "  SphericalShellsInitialRadialRefinement: 2\n"
      "  SphericalShellsInitialRadialGridPoints: 3\n"
      "  InitialSphericalHarmonicL: 6\n"
      "  SphericalShellsRadialPartitioning: []\n"
      "  SphericalShellsRadialDistribution: Linear\n"
      "  Bulge: false\n"
      "  TimeDependentMaps: None\n",
      make_creator(Parameters{}), false);

  TestHelpers::domain::creators::test_creation(
      "BinaryNeutronStars:\n"
      "  XAxisPoints: [-6.0, -3.5, 3.5, 6.0]\n"
      "  WedgeInnerRadius: 2.5\n"
      "  WedgeOuterRadius: 7.0\n"
      "  CylinderOuterRadius: 15.0\n"
      "  OuterRadius: 20.0\n"
      "  CubeInitialGridPoints: [3, 4, 5, 3]\n"
      "  CubeInitialXRefinement: [1, 0, 1, 2]\n"
      "  CubeInitialYZRefinement: [0, 1, 0, 1]\n"
      "  WedgePrismInitialRadialGridPoints: 3\n"
      "  WedgePrismInitialRadialRefinement: 1\n"
      "  CylinderInitialRadialGridPoints: 3\n"
      "  CylinderInitialRadialRefinement: 1\n"
      "  B2InitialAngularGridPoints: 7\n"
      "  HollowCylinderInitialAngularGridPoints: [7, 3]\n"
      "  SphericalShellsInitialRadialRefinement: [1, 2]\n"
      "  SphericalShellsInitialRadialGridPoints: 3\n"
      "  InitialSphericalHarmonicL: 6\n"
      "  SphericalShellsRadialPartitioning: [17.0]\n"
      "  SphericalShellsRadialDistribution: [Linear, Logarithmic]\n"
      "  Bulge: true\n"
      "  TimeDependentMaps: None\n"
      "  OuterBoundary:\n"
      "    TestBoundaryCondition:\n"
      "      Direction: upper-xi\n"
      "      BlockId: 50\n",
      make_creator(equal_mass_parameters(true), std::nullopt,
                   create_outer_boundary_condition()),
      true);

  // A single value for the cube options applies to all four groups
  Parameters same_for_all_groups{};
  same_for_all_groups.cube_grid_points = {4, 4, 4, 4};
  same_for_all_groups.cube_x_refinement = {2, 2, 2, 2};
  same_for_all_groups.cube_yz_refinement = {1, 1, 1, 1};
  TestHelpers::domain::creators::test_creation(
      "BinaryNeutronStars:\n"
      "  XAxisPoints: [-6.9, -4.0, 3.1, 5.8]\n"
      "  WedgeInnerRadius: 2.9\n"
      "  WedgeOuterRadius: 7.0\n"
      "  CylinderOuterRadius: 15.0\n"
      "  OuterRadius: 20.0\n"
      "  CubeInitialGridPoints: 4\n"
      "  CubeInitialXRefinement: 2\n"
      "  CubeInitialYZRefinement: 1\n"
      "  WedgePrismInitialRadialGridPoints: 3\n"
      "  WedgePrismInitialRadialRefinement: 1\n"
      "  CylinderInitialRadialGridPoints: 3\n"
      "  CylinderInitialRadialRefinement: 1\n"
      "  B2InitialAngularGridPoints: 7\n"
      "  HollowCylinderInitialAngularGridPoints: [7, 3]\n"
      "  SphericalShellsInitialRadialRefinement: 2\n"
      "  SphericalShellsInitialRadialGridPoints: 3\n"
      "  InitialSphericalHarmonicL: 6\n"
      "  SphericalShellsRadialPartitioning: []\n"
      "  SphericalShellsRadialDistribution: Linear\n"
      "  Bulge: false\n"
      "  TimeDependentMaps: None\n",
      make_creator(same_for_all_groups), false);
}

void check_parse_error(
    const Parameters& params, const std::string& expected_message,
    std::optional<TimeDependentMapOptions> time_dependent_options =
        std::nullopt,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> outer_bc =
        nullptr) {
  CAPTURE(expected_message);
  CHECK_THROWS_WITH(
      make_creator(params, std::move(time_dependent_options),
                   std::move(outer_bc), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(expected_message));
}

void test_parse_errors() {
  INFO("Test parse errors");
  const std::string ordering_message =
      "must satisfy LeftEnd < CenterB < 0 < CenterA < RightEnd";
  const auto with_x_axis_points = [](const std::array<double, 4>& points) {
    Parameters params{};
    params.x_axis_points = points;
    return params;
  };
  // CenterA not positive
  check_parse_error(with_x_axis_points({-6.9, -4.0, -1.0, 5.8}),
                    ordering_message);
  // CenterB not negative
  check_parse_error(with_x_axis_points({-6.9, 1.0, 3.1, 5.8}),
                    ordering_message);
  // LeftEnd not less than CenterB
  check_parse_error(with_x_axis_points({-3.9, -4.0, 3.1, 5.8}),
                    ordering_message);
  // RightEnd not greater than CenterA
  check_parse_error(with_x_axis_points({-6.9, -4.0, 3.1, 3.0}),
                    ordering_message);
  check_parse_error(with_x_axis_points({-6.9, -4.0, 4.5, 5.8}),
                    "must satisfy CenterA <= |CenterB|, but got CenterA=4.5");

  const auto check_modified = [](const auto& modify,
                                 const std::string& expected_message) {
    Parameters params{};
    modify(params);
    check_parse_error(params, expected_message);
  };
  check_modified([](Parameters& p) { p.wedge_inner_radius = 0.0; },
                 "WedgeInnerRadius must be positive");
  check_modified([](Parameters& p) { p.wedge_outer_radius = -1.0; },
                 "WedgeOuterRadius must be positive");
  check_modified([](Parameters& p) { p.outer_radius = 8.0; },
                 "OuterRadius must be greater than CylinderOuterRadius");
  check_modified([](Parameters& p) { p.cylinder_outer_radius = 4.0; },
                 "CylinderOuterRadius must be greater than WedgeOuterRadius");
  check_modified([](Parameters& p) { p.wedge_outer_radius = 2.0; },
                 "WedgeOuterRadius must be greater than WedgeInnerRadius");
  check_modified(
      [](Parameters& p) {
        p.cylinder_outer_radius = 80.0;
        p.outer_radius = 100.0;
      },
      "CylindricalFlatEndcapInterior radius ratio condition violated");
  // CylinderOuterRadius^2 = 81 <= WedgeOuterRadius^2 + LeftEnd^2 = 97
  check_modified(
      [](Parameters& p) {
        p.x_axis_points = {-9.0, -8.0, 2.0, 3.0};
        p.wedge_inner_radius = 1.0;
        p.wedge_outer_radius = 4.0;
        p.cylinder_outer_radius = 9.0;
      },
      "CylindricalFlatEndcapInterior (the B2s' map) requires");
  check_modified([](Parameters& p) { p.b2_angular_grid_points = 8; },
                 "B2InitialAngularGridPoints must be odd");
  check_modified(
      [](Parameters& p) { p.hollow_cylinder_angular_grid_points = {6, 3}; },
      "HollowCylinderInitialAngularGridPoints[0] must be odd");
  check_modified(
      [](Parameters& p) {
        p.spherical_shells_radial_refinement = std::vector<size_t>{1, 2};
      },
      "The number of radial refinement levels in "
      "SphericalShellsInitialRadialRefinement (2) must match the number of "
      "spherical shells (1)");

  // Bulge=true constraints. Each case passes the checks before the expected
  // one.
  const auto check_bulged =
      [](const std::array<double, 4>& points, const double wedge_inner_radius,
         const double wedge_outer_radius, const double cylinder_outer_radius,
         const std::string& expected_message) {
        Parameters params{};
        params.x_axis_points = points;
        params.wedge_inner_radius = wedge_inner_radius;
        params.wedge_outer_radius = wedge_outer_radius;
        params.cylinder_outer_radius = cylinder_outer_radius;
        params.outer_radius = cylinder_outer_radius + 5.0;
        params.bulge = true;
        check_parse_error(params, expected_message);
      };
  check_bulged({-12.0, -5.5, 2.0, 4.0}, 3.5, 7.0, 16.5,
               "Bulge=true FlatOffsetWedge BLeft constraint");
  check_bulged({-6.5, -3.5, 3.0, 8.0}, 1.5, 5.5, 14.0,
               "Bulge=true FlatOffsetWedge ARight constraint");
  check_bulged({-9.0, -8.0, 3.5, 9.0}, 3.5, 5.0, 16.5,
               "Bulge=true FlatOffsetWedge BRight constraint");
  check_bulged({-7.5, -5.0, 3.5, 8.5}, 4.0, 8.5, 13.0,
               "Bulge=true UniformCylindricalSide BLeft constraint");
  check_bulged({-6.5, -4.0, 1.0, 2.0}, 4.0, 7.0, 18.0,
               "Bulge=true UniformCylindricalSide ARight constraint");
  check_bulged({-6.5, -1.0, 1.0, 6.5}, 4.0, 7.5, 10.0,
               "the far corner of a FlatOffsetWedge block lies outside");
  // The cut between the BLeft and BRight hollow cylinders is too close to x=0
  check_bulged({-2.7, -1.0, 1.0, 2.7}, 1.0, 5.0, 8.0,
               "Bulge=true UniformCylindricalSide BRight outer constraint");
  check_bulged({-6.0, -1.5, 1.5, 4.5}, 2.0, 9.0, 14.0,
               "Bulge=true UniformCylindricalSide ALeft outer constraint");
  // FlatOffsetWedge margins
  check_bulged({-4.3, -4.0, 3.1, 5.8}, 2.9, 7.0, 15.0,
               "the length of the BLeft cubed cylinder along x (0.3");
  check_bulged({-3.5, -2.0, 1.0, 2.5}, 0.5, 4.5, 7.5,
               "Bulge=true FlatOffsetWedge BLeft constraint violated: "
               "WedgeInnerRadius/sqrt(2)");
  check_bulged({-6.9, -4.0, 3.1, 5.8}, 5.3, 7.0, 15.0,
               "Bulge=true FlatOffsetWedge BRight constraint violated: "
               "WedgeInnerRadius^2");

  // RightEnd is less than 5% of CylinderOuterRadius from the origin
  {
    Parameters params{};
    params.x_axis_points = {-3.2, -0.55, 0.5, 2.65};
    params.wedge_inner_radius = 4.8;
    params.wedge_outer_radius = 12.0;
    params.cylinder_outer_radius = 55.0;
    params.outer_radius = 60.0;
    check_parse_error(params,
                      "requires |RightEnd| to be between 0.05 and 0.95 times "
                      "CylinderOuterRadius");
  }

  check_parse_error(
      Parameters{},
      "Cannot have periodic boundary conditions with a binary domain",
      std::nullopt,
      std::make_unique<TestHelpers::domain::BoundaryConditions::
                           TestPeriodicBoundaryCondition<3>>());

  // Shape maps are not supported
  using ShapeMapA = domain::creators::time_dependent_options::ShapeMapOptions<
      false, domain::ObjectLabel::A>;
  check_parse_error(
      Parameters{}, "does not support shape maps",
      TimeDependentMapOptions{0.0, std::nullopt, std::nullopt, std::nullopt,
                              std::nullopt, ShapeMapA{8_st, std::nullopt},
                              std::nullopt, std::nullopt});
}

void test_time_dependent_maps() {
  INFO("Test time-dependent maps");
  const auto make_time_dependent_options = []() {
    return TimeDependentMapOptions{
        0.0,
        domain::creators::time_dependent_options::ExpansionMapOptions<false>{
            std::array{1.0, -1.e-3, 0.0}, 50.0, -1.e-5},
        domain::creators::time_dependent_options::RotationMapOptions<false>{
            std::array{0.0, 0.0, -0.2}},
        domain::creators::time_dependent_options::TranslationMapOptions<3>{
            std::array{std::array{0.0, 0.0, 0.0}, std::array{0.01, 0.0, 0.0},
                       std::array{0.0, 0.0, 0.0}}},
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt};
  };

  const Parameters params{};
  const auto creator = make_creator(params, make_time_dependent_options(),
                                    create_outer_boundary_condition());
  TestHelpers::domain::creators::test_domain_creator(creator, true, false,
                                                     {0.0, 1.0, 5.0});

  const auto domain = creator.create_domain();
  const auto& blocks = domain.blocks();
  REQUIRE(blocks.size() == 27);

  // Reference maps built as in the BinaryNeutronStars constructor
  auto time_dependent_options = make_time_dependent_options();
  time_dependent_options.build_maps(
      std::array{std::array<double, 3>{params.center_A(), 0., 0.},
                 std::array<double, 3>{params.center_B(), 0., 0.}},
      std::nullopt, std::nullopt, std::array<double, 3>{0., 0., 0.},
      std::optional<std::array<double, 2>>{std::nullopt},
      std::optional<std::array<double, 2>>{std::nullopt}, true, true,
      params.cylinder_outer_radius, params.outer_radius);

  for (size_t i = 0; i < blocks.size(); ++i) {
    CAPTURE(i);
    CHECK(blocks[i].is_time_dependent());
    CHECK(not blocks[i].has_distorted_frame());
  }

  // Blocks 0-25 (inside CylinderOuterRadius): rigid map
  const auto rigid_map =
      time_dependent_options.grid_to_inertial_map<domain::ObjectLabel::None>(
          false, true);
  for (size_t i = 0; i < 26; ++i) {
    CAPTURE(i);
    CHECK(blocks[i].moving_mesh_grid_to_inertial_map() == *rigid_map);
  }

  // Block 26 (SphericalShell): transitioning map
  const auto transition_map =
      time_dependent_options.grid_to_inertial_map<domain::ObjectLabel::None>(
          false, false);
  CHECK(blocks[26].moving_mesh_grid_to_inertial_map() == *transition_map);

  const auto functions_of_time = creator.functions_of_time({});
  CHECK(functions_of_time.contains("Expansion"));
  CHECK(functions_of_time.contains("ExpansionOuterBoundary"));
  CHECK(functions_of_time.contains("Rotation"));
  CHECK(functions_of_time.contains("Translation"));
}
}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.BinaryNeutronStars", "[Domain][Unit]") {
  test_block_names_and_groups();
  test_domain_structure(false);
  test_domain_structure(true);
  test_initial_extents_and_refinement();
  test_domain_creator_checks();
  test_geometry();
  test_option_parsing();
  test_parse_errors();
  test_time_dependent_maps();
}
