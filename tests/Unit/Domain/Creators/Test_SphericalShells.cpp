// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/SphericalShells.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/Creators/TimeDependentOptions/Sphere.hpp"
#include "Domain/Creators/TimeDependentOptions/TranslationMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/FillYlmLegendAndData.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Options/Context.hpp"
#include "PointwiseFunctions/GeneralRelativity/KerrHorizon.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_boundary_condition(const bool outer) {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      outer ? Direction<3>::upper_xi() : Direction<3>::lower_xi(), 50);
}

template <typename T>
std::string stringize(const std::vector<T>& t) {
  std::string result = get_output(t).replace(0, 1, "[");
  return result.replace(result.length() - 1, 1, "]");
}

std::string option_string(
    const double inner_radius, const double outer_radius,
    const size_t radial_refinement, const size_t radial_extents,
    const size_t spherical_harmonic_l,
    const std::vector<double>& radial_partitioning,
    const std::vector<domain::CoordinateMaps::Distribution>&
        radial_distribution,
    const bool time_dependent, const bool hard_coded_time_dependent_maps,
    const bool with_boundary_conditions) {
  const std::string time_dependent_option =
      time_dependent ? (hard_coded_time_dependent_maps
                            ? "  TimeDependentMaps:\n"
                              "    InitialTime: 1.0\n"
                              "    ShapeMap:\n"
                              "      LMax: 10\n"
                              "      InitialValues: Spherical\n"
                              "      CoefficientTruncationLimit: 0.0\n"
                              "      SizeInitialValues: Auto\n"
                              "    RotationMap: None\n"
                              "    ExpansionMap: None\n"
                              "    TranslationMap:\n"
                              "      InitialValues: [[0.0, 0.0, 0.0],"
                              " [0.001, -0.003, 0.005], [0.0, 0.0, 0.0]]\n"
                              "    TransitionRotScaleTrans: False\n"
                            : "  TimeDependentMaps:\n"
                              "    UniformTranslation:\n"
                              "      InitialTime: 1.0\n"
                              "      Velocity: [2.3, -0.3, 0.5]\n")
                     : "  TimeDependentMaps: None\n";
  const std::string inner_bc_option = with_boundary_conditions
                                          ? "  InnerBoundaryCondition:\n"
                                            "    TestBoundaryCondition:\n"
                                            "      Direction: lower-xi\n"
                                            "      BlockId: 50\n"
                                          : "";
  const std::string outer_bc_option = with_boundary_conditions
                                          ? "  OuterBoundaryCondition:\n"
                                            "    TestBoundaryCondition:\n"
                                            "      Direction: upper-xi\n"
                                            "      BlockId: 50\n"
                                          : "";
  return "SphericalShells:\n"
         "  InnerRadius: " +
         std::to_string(inner_radius) +
         "\n"
         "  OuterRadius: " +
         std::to_string(outer_radius) + "\n" +
         "  InitialRadialRefinement: " + std::to_string(radial_refinement) +
         "\n"
         "  InitialNumberOfRadialGridPoints: " +
         std::to_string(radial_extents) +
         "\n"
         "  InitialSphericalHarmonicL: " +
         std::to_string(spherical_harmonic_l) +
         "\n"
         "  RadialPartitioning: " +
         stringize(radial_partitioning) +
         "\n"
         "  RadialDistribution: " +
         (radial_distribution.size() == 1 ? get_output(radial_distribution[0])
                                          : stringize(radial_distribution)) +
         "\n" + time_dependent_option + inner_bc_option + outer_bc_option;
}

void test_parse_errors() {
  INFO("SphericalShells check throws");
  const double inner_radius = 1.0;
  const double outer_radius = 2.0;
  const size_t radial_refinement = 2;
  const size_t radial_extents = 5;
  const size_t l = 6;
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
      domain::creators::SphericalShells(
          inner_radius, 0.5 * inner_radius, radial_refinement, radial_extents,
          l, radial_partitioning, radial_distribution, std::nullopt, nullptr,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Inner radius must be smaller than outer radius"));

  CHECK_THROWS_WITH(
      domain::creators::SphericalShells(
          inner_radius, outer_radius, radial_refinement, radial_extents, l,
          radial_partitioning_unordered, radial_distribution, std::nullopt,
          nullptr, nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Specify radial partitioning in ascending order."));

  CHECK_THROWS_WITH(
      domain::creators::SphericalShells(
          inner_radius, outer_radius, radial_refinement, radial_extents, l,
          radial_partitioning_low, radial_distribution, std::nullopt, nullptr,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "First radial partition must be larger than the inner"));
  CHECK_THROWS_WITH(
      domain::creators::SphericalShells(
          inner_radius, outer_radius, radial_refinement, radial_extents, l,
          radial_partitioning_high, radial_distribution, std::nullopt, nullptr,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Last radial partition must be smaller than the outer"));
  CHECK_THROWS_WITH(
      domain::creators::SphericalShells(
          inner_radius, outer_radius, radial_refinement, radial_extents, l,
          radial_partitioning, radial_distribution_too_many, std::nullopt,
          nullptr, nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Specify a 'RadialDistribution' for every spherical shell. You"));

  CHECK_THROWS_WITH(
      domain::creators::SphericalShells(
          inner_radius, outer_radius, radial_refinement, radial_extents, l,
          radial_partitioning, radial_distribution, std::nullopt,
          create_boundary_condition(false), nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Must specify either both inner and outer boundary conditions "
          "or neither."));
  CHECK_THROWS_WITH(
      domain::creators::SphericalShells(
          inner_radius, outer_radius, radial_refinement, radial_extents, l,
          radial_partitioning, radial_distribution, std::nullopt,
          create_boundary_condition(false),
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Cannot have periodic boundary conditions with SphericalShells"));
  CHECK_THROWS_WITH(
      domain::creators::SphericalShells(
          inner_radius, outer_radius, radial_refinement, radial_extents, l,
          radial_partitioning, radial_distribution, std::nullopt,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          create_boundary_condition(true), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Cannot have periodic boundary conditions with SphericalShells"));
  CHECK_THROWS_WITH(
      domain::creators::SphericalShells(
          inner_radius, outer_radius, radial_refinement, radial_extents, l,
          radial_partitioning, radial_distribution, std::nullopt,
          create_boundary_condition(false),
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "None boundary condition is not supported. If you would like "
          "an outflow-type boundary condition, you must use that."));
  CHECK_THROWS_WITH(
      domain::creators::SphericalShells(
          inner_radius, outer_radius, radial_refinement, radial_extents, l,
          radial_partitioning, radial_distribution, std::nullopt,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          create_boundary_condition(true), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "None boundary condition is not supported. If you would like "
          "an outflow-type boundary condition, you must use that."));
}

template <typename Generator>
void test_spherical_shells_construction(
    const gsl::not_null<Generator*> gen,
    const domain::creators::SphericalShells& spherical_shells,
    const double inner_radius, const double outer_radius,
    const std::vector<double>& radial_partitioning = {},
    const bool expect_boundary_conditions = true,
    const std::vector<double>& times = {0.},
    const std::array<double, 3>& velocity = {{0., 0., 0.}}) {
  // check consistency of domain
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      spherical_shells, expect_boundary_conditions, false, times);
  const auto& grid_anchors = spherical_shells.grid_anchors();
  CHECK(grid_anchors.size() == 1);
  CHECK(grid_anchors.count("Center") == 1);
  CHECK(grid_anchors.at("Center") ==
        tnsr::I<double, 3, Frame::Grid>{std::array{0.0, 0.0, 0.0}});

  const auto& blocks = domain.blocks();
  const auto block_names = spherical_shells.block_names();
  const size_t num_blocks = blocks.size();
  CAPTURE(num_blocks);
  const auto all_boundary_conditions =
      spherical_shells.external_boundary_conditions();
  const auto functions_of_time = spherical_shells.functions_of_time();

  // Check total number of external boundaries
  const size_t num_shells = radial_partitioning.size() + 1;
  CHECK(num_blocks == num_shells);
  const size_t num_external_boundaries =
      alg::accumulate(blocks, 0_st, [](const size_t count, const auto& block) {
        return count + block.external_boundaries().size();
      });
  CHECK(num_external_boundaries == 2);

  std::vector<double> expected_radii = radial_partitioning;
  expected_radii.insert(expected_radii.begin(), inner_radius);
  expected_radii.emplace_back(outer_radius);

  // NOLINTNEXTLINE(misc-const-correctness)
  std::uniform_real_distribution<> theta_distribution(0.0, M_PI);
  // NOLINTNEXTLINE(misc-const-correctness)
  std::uniform_real_distribution<> phi_distribution(0.0, 2.0 * M_PI);
  for (size_t block_id = 0; block_id < num_blocks; ++block_id) {
    CAPTURE(block_id);
    const auto& block = blocks[block_id];
    const ElementMap<3, Frame::Grid> grid_element_map{ElementId<3>{block_id},
                                                      block};
    const ElementMap<3, Frame::Inertial> inertial_element_map{
        ElementId<3>{block_id}, block};
    {
      INFO("Radius of random point on lower face");
      const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
          {{-1.0, theta_distribution(*gen), phi_distribution(*gen)}}};
      for (const double current_time : times) {
        CAPTURE(current_time);
        auto x_inertial =
            inertial_element_map(x_logical, current_time, functions_of_time);
        const double delta_t = current_time - 1.0;
        for (size_t i = 0; i < 3; ++i) {
          x_inertial.get(i) -= gsl::at(velocity, i) * delta_t;
        }
        CHECK(get(magnitude(x_inertial)) == approx(expected_radii[block_id]));
      }
    }
    {
      INFO("Radius of random point on upper face");
      const double r = expected_radii[block_id + 1];
      const double theta = theta_distribution(*gen);
      const double phi = phi_distribution(*gen);
      const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
          {{1.0, theta, phi}}};
      const tnsr::I<double, 3, Frame::Grid> expected_x_grid{
          {{r * sin(theta) * cos(phi), r * sin(theta) * sin(phi),
            r * cos(theta)}}};
      for (const double current_time : times) {
        CAPTURE(current_time);
        const auto x_grid =
            grid_element_map(x_logical, current_time, functions_of_time);
        CAPTURE(x_grid);
        CHECK_ITERABLE_APPROX(x_grid, expected_x_grid);
        auto x_inertial =
            inertial_element_map(x_logical, current_time, functions_of_time);
        const double delta_t = current_time - 1.0;
        for (size_t i = 0; i < 3; ++i) {
          x_inertial.get(i) -= gsl::at(velocity, i) * delta_t;
        }
        CHECK(get(magnitude(x_inertial)) ==
              approx(expected_radii[block_id + 1]));
      }
    }
    {
      INFO("External boundaries");
      const auto& external_boundaries = block.external_boundaries();
      if (num_blocks == 1) {
        CHECK(external_boundaries.size() == 2);
        CHECK(alg::found(external_boundaries, Direction<3>::lower_xi()));
        CHECK(alg::found(external_boundaries, Direction<3>::upper_xi()));
      } else if (block_id == 0) {
        CHECK(external_boundaries.size() == 1);
        CHECK(alg::found(external_boundaries, Direction<3>::lower_xi()));
      } else if (block_id == num_blocks - 1) {
        CHECK(external_boundaries.size() == 1);
        CHECK(alg::found(external_boundaries, Direction<3>::upper_xi()));
      } else {
        CHECK(external_boundaries.empty());
      }
    }
    if (expect_boundary_conditions) {
      INFO("Boundary conditions");
      const auto& boundary_conditions = all_boundary_conditions[block_id];
      for (const auto& direction : block.external_boundaries()) {
        CAPTURE(direction);
        const auto& boundary_condition =
            dynamic_cast<const TestHelpers::domain::BoundaryConditions::
                             TestBoundaryCondition<3>&>(
                *boundary_conditions.at(direction));
        CHECK(boundary_condition.direction() == direction);
      }
    }
  }
}

template <typename Generator>
void test_sphere(const gsl::not_null<Generator*> gen) {
  const double inner_radius = 1.0;
  const double outer_radius = 2.0;
  const size_t radial_refinement = 3;
  const size_t radial_extents = 5;
  const size_t l = 6;
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
  const size_t l_max = 10;
  const std::vector<double> times{1., 10.};
  for (auto [index, time_dependent, use_hard_coded_time_dep_options,
             with_boundary_conditions] :
       random_sample<5>(
           cartesian_product(make_array(0_st, 1_st, 2_st),
                             make_array(true, false), make_array(true, false),
                             make_array(true, false)),
           gen)) {
    CAPTURE(time_dependent);
    CAPTURE(with_boundary_conditions);
    // If we aren't time dependent, just set the hard coded option to false to
    // avoid ambiguity
    if (not time_dependent) {
      use_hard_coded_time_dep_options = false;
    }
    CAPTURE(use_hard_coded_time_dep_options);
    // If we are using hard coded maps, we need at least two shells (or one
    // radial partition) for the translation map.
    auto array_index = (use_hard_coded_time_dep_options and
                                gsl::at(radial_partitioning, index).empty()
                            ? index + 1
                            : index);
    CAPTURE(gsl::at(radial_partitioning, array_index));
    CAPTURE(gsl::at(radial_distribution, array_index));
    domain::creators::SphericalShells::RadialDistribution::type
        radial_distribution_variant;
    if (gsl::at(radial_distribution, array_index).size() == 1) {
      radial_distribution_variant =
          gsl::at(radial_distribution, array_index)[0];
    } else {
      radial_distribution_variant = gsl::at(radial_distribution, array_index);
    }

    std::optional<domain::creators::SphericalShells::TimeDepOptionType>
        time_dependent_options{};

    auto translation_velocity = std::array<double, 3>{{0., 0., 0.}};

    if (time_dependent) {
      if (use_hard_coded_time_dep_options) {
        using namespace domain::creators::time_dependent_options;  // NOLINT
        translation_velocity = std::array<double, 3>{{0.001, -0.003, 0.005}};
        time_dependent_options =
            domain::creators::sphere::TimeDependentMapOptions{
                1.0,
                ShapeMapOptions<false, domain::ObjectLabel::None>{l_max,
                                                                  std::nullopt},
                std::nullopt,
                std::nullopt,
                TranslationMapOptions<3>{std::array{
                    std::array<double, 3>{0.0, 0.0, 0.0}, translation_velocity,
                    std::array<double, 3>{0.0, 0.0, 0.0}}},
                false};
      } else {
        time_dependent_options = std::make_unique<
            domain::creators::time_dependence::UniformTranslation<3, 0>>(
            1.0, velocity);
        translation_velocity = velocity;
      }
    }

    const domain::creators::SphericalShells spherical_shells{
        inner_radius,
        outer_radius,
        radial_refinement,
        radial_extents,
        l,
        gsl::at(radial_partitioning, array_index),
        radial_distribution_variant,
        std::move(time_dependent_options),
        with_boundary_conditions ? create_boundary_condition(false) : nullptr,
        with_boundary_conditions ? create_boundary_condition(true) : nullptr};
    test_spherical_shells_construction(
        gen, spherical_shells, inner_radius, outer_radius,
        gsl::at(radial_partitioning, array_index), with_boundary_conditions,
        time_dependent ? times : std::vector<double>{1.}, translation_velocity);
    TestHelpers::domain::creators::test_creation(
        option_string(
            inner_radius, outer_radius, radial_refinement, radial_extents, l,
            gsl::at(radial_partitioning, array_index),
            gsl::at(radial_distribution, array_index), time_dependent,
            use_hard_coded_time_dep_options, with_boundary_conditions),
        spherical_shells, with_boundary_conditions);
  }
}

void test_shape_distortion_general(
    const double time,
    domain::creators::SphericalShells::TimeDepOptionType time_dependent_options,
    const double deformed_radius, const tnsr::I<DataVector, 3>& x) {
  const domain::creators::SphericalShells domain_creator{
      deformed_radius,
      10.,
      0_st,
      5_st,
      6_st,
      {4.},
      domain::CoordinateMaps::Distribution::Linear,
      std::move(time_dependent_options)};
  const auto domain = domain_creator.domain();
  const auto functions_of_time = domain_creator.functions_of_time();
  // Map the coordinates through the domain. They should lie at the lower xi
  // boundary of their block.
  const auto x_logical =
      block_logical_coordinates(domain, x, time, functions_of_time);
  for (size_t i = 0; i < get<0>(x).size(); ++i) {
    CAPTURE(x_logical[i]);
    REQUIRE(x_logical[i].has_value());
    CHECK(get<0>(x_logical[i]->data) == approx(-1.));
  }
}

void test_shape_distortion() {
  domain::creators::SphericalShells::TimeDepOptionType time_dependent_options{};

  // Set up theta phis
  const size_t l_max = 16;
  const ylm::Spherepack ylm{l_max, l_max};
  const std::array<DataVector, 2> theta_phi = ylm.theta_phi_points();

  const double time = 0.7;
  const double mass = 0.8;
  const std::array<double, 3> spin{{0.0, 0.0, 0.9}};
  const double r_plus = mass * (1. + sqrt(1. - dot(spin, spin)));
  const double inner_radius = r_plus;

  const DataVector radius =
      get(gr::Solutions::kerr_schild_radius_from_boyer_lindquist(
          inner_radius, theta_phi, mass, spin));
  // Set up coordinates on an ellipsoid of constant Boyer-Lindquist radius
  tnsr::I<DataVector, 3> x{};
  get<0>(x) = radius * sin(get<0>(theta_phi)) * cos(get<1>(theta_phi));
  get<1>(x) = radius * sin(get<0>(theta_phi)) * sin(get<1>(theta_phi));
  get<2>(x) = radius * cos(get<0>(theta_phi));
  {
    INFO(
        "Check that inner radius is deformed to constant Boyer-Lindquist "
        "radius");

    // Time dependence
    time_dependent_options = std::make_unique<
        domain::creators::time_dependence::Shape<domain::ObjectLabel::None>>(
        time, l_max, mass, spin, std::array<double, 3>{{0., 0., 0.}}, 0.0,
        inner_radius, 4.);

    test_shape_distortion_general(time, std::move(time_dependent_options),
                                  inner_radius, x);

    // KerrSchild-BoyerLindquist. Use same x as above
    time_dependent_options = domain::creators::sphere::TimeDependentMapOptions{
        time,
        domain::creators::time_dependent_options::ShapeMapOptions<
            false, domain::ObjectLabel::None>{
            l_max, domain::creators::time_dependent_options::
                       KerrSchildFromBoyerLindquist{mass, spin}},
        std::nullopt,
        std::nullopt,
        std::nullopt,
        true};
    test_shape_distortion_general(time, std::move(time_dependent_options),
                                  inner_radius, x);
  }

  {
    INFO("Check reading in Ylms from file");
    const std::string h5_filename{"S2_StrahlkorperCoefsFile.h5"};
    const std::string subfile_name{"Ylm_coefs"};
    if (file_system::check_if_file_exists(h5_filename)) {
      file_system::rm(h5_filename, true);
    }
    const ylm::Strahlkorper<Frame::Distorted> strahlkorper{
        l_max, l_max, radius, std::array{0.0, 0.0, 0.0}};
    std::vector<std::string> legend{};
    std::vector<double> data{};
    ylm::fill_ylm_legend_and_data(make_not_null(&legend), make_not_null(&data),
                                  strahlkorper, time, l_max);
    {
      h5::H5File<h5::AccessType::ReadWrite> test_file{h5_filename, true};
      auto& subfile = test_file.insert<h5::Dat>("/" + subfile_name, legend);
      subfile.append(data);
      test_file.close_current_object();
    }

    time_dependent_options = domain::creators::sphere::TimeDependentMapOptions{
        time,
        domain::creators::time_dependent_options::ShapeMapOptions<
            false, domain::ObjectLabel::None>{
            l_max,
            domain::creators::time_dependent_options::YlmsFromFile{
                h5_filename, std::vector{subfile_name}, time, std::nullopt,
                false},
            // Constructing a strahlkorper from collocation radii will not
            // exactly match the collocation points (see Strahlkorper
            // constructor docs). For this reason, the 00 coef we calculate is
            // not exact. This is why we just hard code the proper value here.
            // If you change l_max, this value must also change.
            std::array{-4.6442771561420703730e-01, 0.0, 0.0}},
        std::nullopt,
        std::nullopt,
        std::nullopt,
        true};

    test_shape_distortion_general(time, std::move(time_dependent_options),
                                  inner_radius, x);

    if (file_system::check_if_file_exists(h5_filename)) {
      file_system::rm(h5_filename, true);
    }
  }
}
}  // namespace

// [[TimeOut, 15]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.SphericalShells", "[Domain][Unit]") {
  MAKE_GENERATOR(gen);
  domain::creators::time_dependence::register_derived_with_charm();
  test_parse_errors();
  test_sphere(make_not_null(&gen));
  test_shape_distortion();
}
