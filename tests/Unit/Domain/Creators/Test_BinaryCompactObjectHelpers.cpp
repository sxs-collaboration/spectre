// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/BinaryCompactObjectHelpers.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Grid;
struct Distorted;
struct Inertial;
}  // namespace Frame

namespace domain::creators::bco {
using ExpMapOptions = TimeDependentMapOptions::ExpansionMapOptions;
using RotMapOptions = TimeDependentMapOptions::RotationMapOptions;
using ShapeMapAOptions =
    TimeDependentMapOptions::ShapeMapOptions<domain::ObjectLabel::A>;
using ShapeMapBOptions =
    TimeDependentMapOptions::ShapeMapOptions<domain::ObjectLabel::B>;
namespace {
// Test produce_all_maps for 1-4 maps
using Expansion = domain::CoordinateMaps::TimeDependent::CubicScale<3>;
using Rotation = domain::CoordinateMaps::TimeDependent::Rotation<3>;
using Shape = domain::CoordinateMaps::TimeDependent::Shape;
using Identity = domain::CoordinateMaps::Identity<3>;

using combo_maps_1 =
    detail::produce_all_maps<Frame::Grid, Frame::Inertial, Expansion>;
using combo_maps_1_expected = tmpl::list<detail::gi_map<Expansion>>;
using combo_maps_2 =
    detail::produce_all_maps<Frame::Grid, Frame::Inertial, Expansion, Rotation>;
using combo_maps_2_expected =
    // clang-format off
    tmpl::list<detail::gi_map<Expansion>,
               detail::gi_map<Rotation>,
               detail::gi_map<Expansion, Rotation>>;
// clang-format on
using combo_maps_3 = detail::produce_all_maps<Frame::Grid, Frame::Inertial,
                                              Shape, Expansion, Rotation>;
using combo_maps_3_expected =
    // clang-format off
    tmpl::list<detail::gi_map<Expansion>,
               detail::gi_map<Rotation>,
               detail::gi_map<Shape>,
               detail::gi_map<Shape, Expansion>,
               detail::gi_map<Shape, Rotation>,
               detail::gi_map<Expansion, Rotation>,
               detail::gi_map<Shape, Expansion, Rotation>>;
// clang-format on
using combo_maps_4 =
    detail::produce_all_maps<Frame::Grid, Frame::Inertial, Shape, Expansion,
                             Rotation, Identity>;
using combo_maps_4_expected =
    // clang-format off
    tmpl::list<detail::gi_map<Expansion>,
               detail::gi_map<Rotation>,
               detail::gi_map<Shape>,
               detail::gi_map<Identity>,
               detail::gi_map<Shape, Expansion>,
               detail::gi_map<Shape, Rotation>,
               detail::gi_map<Shape, Identity>,
               detail::gi_map<Expansion, Rotation>,
               detail::gi_map<Expansion, Identity>,
               detail::gi_map<Rotation, Identity>,
               detail::gi_map<Shape, Expansion, Rotation>,
               detail::gi_map<Shape, Expansion, Identity>,
               detail::gi_map<Shape, Rotation, Identity>,
               detail::gi_map<Expansion, Rotation, Identity>,
               detail::gi_map<Shape, Expansion, Rotation, Identity>>;
// clang-format on

template <typename List1, typename List2>
constexpr bool check_maps_list_v =
    std::is_same_v<tmpl::list_difference<List1, List2>, tmpl::list<>>;

static_assert(check_maps_list_v<combo_maps_1, combo_maps_1_expected>);
static_assert(check_maps_list_v<combo_maps_1_expected, combo_maps_1>);
static_assert(check_maps_list_v<combo_maps_2, combo_maps_2_expected>);
static_assert(check_maps_list_v<combo_maps_2_expected, combo_maps_2>);
static_assert(check_maps_list_v<combo_maps_3, combo_maps_3_expected>);
static_assert(check_maps_list_v<combo_maps_3_expected, combo_maps_3>);
static_assert(check_maps_list_v<combo_maps_4, combo_maps_4_expected>);
static_assert(check_maps_list_v<combo_maps_4_expected, combo_maps_4>);

void test(const bool include_expansion, const bool include_rotation,
          const bool include_shape_a, const bool include_shape_b) {
  CAPTURE(include_expansion);
  CAPTURE(include_rotation);
  CAPTURE(include_shape_a);
  CAPTURE(include_shape_b);
  std::optional<ExpMapOptions> exp_map_options{};
  std::optional<RotMapOptions> rot_map_options{};
  std::optional<ShapeMapAOptions> shape_map_a_options{};
  std::optional<ShapeMapBOptions> shape_map_b_options{};

  const std::array<double, 2> exp_values{1.0, 0.0};
  const double exp_outer_boundary_velocity = -0.01;
  const double exp_outer_boundary_timescale = 25.0;
  if (include_expansion) {
    exp_map_options = ExpMapOptions{exp_values, exp_outer_boundary_velocity,
                                    exp_outer_boundary_timescale};
  }

  const std::array<double, 3> angular_velocity{0.2, -0.4, 0.6};
  if (include_rotation) {
    rot_map_options = RotMapOptions{angular_velocity};
  }

  const std::array<double, 3> size_A_values{0.9, 0.08, 0.007};
  const size_t l_max_A = 8;
  if (include_shape_a) {
    shape_map_a_options = ShapeMapAOptions{l_max_A, size_A_values};
  }

  const std::array<double, 3> size_B_values{-0.001, -0.02, -0.3};
  const size_t l_max_B = 10;
  if (include_shape_b) {
    shape_map_b_options = ShapeMapBOptions{l_max_B, size_B_values};
  }

  const double initial_time = 1.5;

  if ((not include_expansion) and (not include_rotation) and
      (not include_shape_a) and (not include_shape_b)) {
    CHECK_THROWS_WITH(
        (TimeDependentMapOptions{initial_time, exp_map_options, rot_map_options,
                                 shape_map_a_options, shape_map_b_options}),
        Catch::Contains(
            "Time dependent map options were specified, but all options "
            "were 'None'. If you don't want time dependent maps, specify "
            "'None' for the TimeDependentMapOptions. If you want time "
            "dependent maps, specify options for at least one map."));
    return;
  }

  TimeDependentMapOptions time_dep_options{initial_time, exp_map_options,
                                           rot_map_options, shape_map_a_options,
                                           shape_map_b_options};

  CHECK(time_dep_options.has_distorted_frame_options(domain::ObjectLabel::A) ==
        include_shape_a);
  CHECK(time_dep_options.has_distorted_frame_options(domain::ObjectLabel::B) ==
        include_shape_b);

  std::unordered_map<std::string, double> expiration_times{
      {TimeDependentMapOptions::expansion_name, 10.0},
      {TimeDependentMapOptions::rotation_name,
       std::numeric_limits<double>::infinity()},
      {gsl::at(TimeDependentMapOptions::size_names, 0), 15.5},
      {gsl::at(TimeDependentMapOptions::size_names, 1),
       std::numeric_limits<double>::infinity()},
      {gsl::at(TimeDependentMapOptions::shape_names, 0),
       std::numeric_limits<double>::infinity()},
      {gsl::at(TimeDependentMapOptions::shape_names, 1), 19.1}};

  using ExpFoT = domain::FunctionsOfTime::PiecewisePolynomial<2>;
  using ExpBdryFoT = domain::FunctionsOfTime::FixedSpeedCubic;
  using RotFoT = domain::FunctionsOfTime::QuaternionFunctionOfTime<3>;
  using SizeFoT = domain::FunctionsOfTime::PiecewisePolynomial<3>;
  using ShapeFoT = ExpFoT;
  ExpFoT expansion{
      initial_time,
      std::array<DataVector, 3>{
          {{gsl::at(exp_values, 0)}, {gsl::at(exp_values, 1)}, {0.0}}},
      expiration_times.at(TimeDependentMapOptions::expansion_name)};
  ExpBdryFoT expansion_outer_boundary{1.0, initial_time,
                                      exp_outer_boundary_velocity,
                                      exp_outer_boundary_timescale};
  RotFoT rotation{initial_time,
                  std::array<DataVector, 1>{DataVector{1.0, 0.0, 0.0, 0.0}},
                  std::array<DataVector, 4>{{{3, 0.0},
                                             {gsl::at(angular_velocity, 0),
                                              gsl::at(angular_velocity, 1),
                                              gsl::at(angular_velocity, 2)},
                                             {3, 0.0},
                                             {3, 0.0}}},
                  expiration_times.at(TimeDependentMapOptions::rotation_name)};
  SizeFoT size_A{
      initial_time,
      std::array<DataVector, 4>{{{gsl::at(size_A_values, 0)},
                                 {gsl::at(size_A_values, 1)},
                                 {gsl::at(size_A_values, 2)},
                                 {0.0}}},
      expiration_times.at(gsl::at(TimeDependentMapOptions::size_names, 0))};
  SizeFoT size_B{
      initial_time,
      std::array<DataVector, 4>{{{gsl::at(size_B_values, 0)},
                                 {gsl::at(size_B_values, 1)},
                                 {gsl::at(size_B_values, 2)},
                                 {0.0}}},
      expiration_times.at(gsl::at(TimeDependentMapOptions::size_names, 1))};
  const DataVector shape_A_zeros{
      ylm::Spherepack::spectral_size(l_max_A, l_max_A), 0.0};
  const DataVector shape_B_zeros{
      ylm::Spherepack::spectral_size(l_max_B, l_max_B), 0.0};
  ShapeFoT shape_A{
      initial_time,
      std::array<DataVector, 3>{shape_A_zeros, shape_A_zeros, shape_A_zeros},
      expiration_times.at(gsl::at(TimeDependentMapOptions::shape_names, 0))};
  ShapeFoT shape_B{
      initial_time,
      std::array<DataVector, 3>{shape_B_zeros, shape_B_zeros, shape_B_zeros},
      expiration_times.at(gsl::at(TimeDependentMapOptions::shape_names, 1))};

  const auto functions_of_time =
      time_dep_options.create_functions_of_time(expiration_times);

  if (include_expansion) {
    CHECK(functions_of_time.count(TimeDependentMapOptions::expansion_name) ==
          1);
    CHECK(functions_of_time.count(
              TimeDependentMapOptions::expansion_outer_boundary_name) == 1);
    CHECK(dynamic_cast<ExpFoT&>(
              *functions_of_time.at(TimeDependentMapOptions::expansion_name)
                   .get()) == expansion);
    CHECK(dynamic_cast<ExpBdryFoT&>(
              *functions_of_time
                   .at(TimeDependentMapOptions::expansion_outer_boundary_name)
                   .get()) == expansion_outer_boundary);
  } else {
    CHECK(functions_of_time.count(TimeDependentMapOptions::expansion_name) ==
          0);
    CHECK(functions_of_time.count(
              TimeDependentMapOptions::expansion_outer_boundary_name) == 0);
  }
  if (include_rotation) {
    CHECK(functions_of_time.count(TimeDependentMapOptions::rotation_name) == 1);
    CHECK(dynamic_cast<RotFoT&>(
              *functions_of_time.at(TimeDependentMapOptions::rotation_name)
                   .get()) == rotation);
  } else {
    CHECK(functions_of_time.count(TimeDependentMapOptions::rotation_name) == 0);
  }
  if (include_shape_a) {
    CHECK(functions_of_time.count(
              gsl::at(TimeDependentMapOptions::size_names, 0)) == 1);
    CHECK(functions_of_time.count(
              gsl::at(TimeDependentMapOptions::shape_names, 0)) == 1);
    CHECK(dynamic_cast<SizeFoT&>(
              *functions_of_time
                   .at(gsl::at(TimeDependentMapOptions::size_names, 0))
                   .get()) == size_A);
    CHECK(dynamic_cast<ShapeFoT&>(
              *functions_of_time
                   .at(gsl::at(TimeDependentMapOptions::shape_names, 0))
                   .get()) == shape_A);
  } else {
    CHECK(functions_of_time.count(
              gsl::at(TimeDependentMapOptions::size_names, 0)) == 0);
    CHECK(functions_of_time.count(
              gsl::at(TimeDependentMapOptions::shape_names, 0)) == 0);
  }
  if (include_shape_b) {
    CHECK(functions_of_time.count(
              gsl::at(TimeDependentMapOptions::size_names, 1)) == 1);
    CHECK(functions_of_time.count(
              gsl::at(TimeDependentMapOptions::shape_names, 1)) == 1);
    CHECK(dynamic_cast<SizeFoT&>(
              *functions_of_time
                   .at(gsl::at(TimeDependentMapOptions::size_names, 1))
                   .get()) == size_B);
    CHECK(dynamic_cast<ShapeFoT&>(
              *functions_of_time
                   .at(gsl::at(TimeDependentMapOptions::shape_names, 1))
                   .get()) == shape_B);
  } else {
    CHECK(functions_of_time.count(
              gsl::at(TimeDependentMapOptions::size_names, 1)) == 0);
    CHECK(functions_of_time.count(
              gsl::at(TimeDependentMapOptions::shape_names, 1)) == 0);
  }

  const std::array<std::array<double, 3>, 2> centers{
      std::array{5.0, 0.01, 0.02}, std::array{-5.0, -0.01, -0.02}};
  const double domain_outer_radius = 20.0;

  const auto anchors = create_grid_anchors(centers[0], centers[1]);
  CHECK(anchors.count("Center") == 1);
  CHECK(anchors.count("CenterA") == 1);
  CHECK(anchors.count("CenterB") == 1);
  CHECK(anchors.at("Center") ==
        tnsr::I<double, 3, Frame::Grid>(std::array{0.0, 0.0, 0.0}));
  CHECK(anchors.at("CenterA") == tnsr::I<double, 3, Frame::Grid>(centers[0]));
  CHECK(anchors.at("CenterB") == tnsr::I<double, 3, Frame::Grid>(centers[1]));

  for (const auto& [excise_A, excise_B] :
       cartesian_product(make_array(true, false), make_array(true, false))) {
    CAPTURE(excise_A);
    CAPTURE(excise_B);
    std::optional<std::pair<double, double>> inner_outer_radii_A{};
    std::optional<std::pair<double, double>> inner_outer_radii_B{};
    if (excise_A) {
      inner_outer_radii_A = std::make_pair(0.8, 3.2);
    }
    if (excise_B) {
      inner_outer_radii_B = std::make_pair(0.5, 2.1);
    }

    const auto build_maps = [&time_dep_options, &centers, &inner_outer_radii_A,
                             &inner_outer_radii_B, &domain_outer_radius]() {
      time_dep_options.build_maps(centers, inner_outer_radii_A,
                                  inner_outer_radii_B, domain_outer_radius);
    };

    // If we have excision info, but didn't specify shape map options, we should
    // hit an error.
    if (excise_A and not include_shape_a) {
      CHECK_THROWS_WITH(
          build_maps(),
          Catch::Contains("Trying to build the shape map for object"));
      continue;
    }
    // If we have shape map options, but didn't specify excisions, we should hit
    // an error.
    if (include_shape_a and not excise_A) {
      CHECK_THROWS_WITH(
          build_maps(),
          Catch::Contains("No excision was specified for object"));
      continue;
    }

    // Same for B
    if (excise_B and not include_shape_b) {
      CHECK_THROWS_WITH(
          build_maps(),
          Catch::Contains("Trying to build the shape map for object"));
      continue;
    }
    if (include_shape_b and not excise_B) {
      CHECK_THROWS_WITH(
          build_maps(),
          Catch::Contains("No excision was specified for object"));
      continue;
    }

    // Now for each object, either we have included shape map options and
    // excision info, or we didn't include either. (i.e. include_shape_map_? and
    // excise_? are both true or both false) so it's safe to build the maps now
    build_maps();

    if ((not include_rotation) and (not include_expansion) and (not excise_A)) {
      CHECK_THROWS_WITH(
          time_dep_options.grid_to_inertial_map<domain::ObjectLabel::A>(
              excise_A),
          Catch::Contains(
              "Requesting grid to inertial map without a distorted frame and "
              "without a Rotation or Expansion map for object"));
      continue;
    }
    if ((not include_rotation) and (not include_expansion) and (not excise_B)) {
      CHECK_THROWS_WITH(
          time_dep_options.grid_to_inertial_map<domain::ObjectLabel::B>(
              excise_B),
          Catch::Contains(
              "Requesting grid to inertial map without a distorted frame and "
              "without a Rotation or Expansion map for object"));
      continue;
    }

    const auto grid_to_distorted_map_A =
        time_dep_options.grid_to_distorted_map<domain::ObjectLabel::A>(
            excise_A);
    const auto grid_to_distorted_map_B =
        time_dep_options.grid_to_distorted_map<domain::ObjectLabel::B>(
            excise_B);
    const auto grid_to_inertial_map_A =
        time_dep_options.grid_to_inertial_map<domain::ObjectLabel::A>(excise_A);
    const auto grid_to_inertial_map_B =
        time_dep_options.grid_to_inertial_map<domain::ObjectLabel::B>(excise_B);
    // Even though the distorted to inertial map is not tied to a specific
    // object, we use `excise_?` to determine if the distorted map is
    // included just for testing.
    const auto distorted_to_inertial_map_A =
        time_dep_options.distorted_to_inertial_map(excise_A);
    const auto distorted_to_inertial_map_B =
        time_dep_options.distorted_to_inertial_map(excise_B);

    // All of these maps are tested individually. Rather than going through the
    // effort of coming up with a source coordinate and calculating analytically
    // what we would get after it's mapped, we just check whether it's supposed
    // to be a nullptr and if it's not that it's not the identity and that the
    // jacobians are time dependent.
    const auto check_map = [](const auto& map, const bool is_null,
                              const bool is_identity) {
      if (is_null) {
        CHECK(map == nullptr);
      } else {
        CHECK(map->is_identity() == is_identity);
        CHECK(map->inv_jacobian_is_time_dependent() != is_identity);
        CHECK(map->jacobian_is_time_dependent() != is_identity);
      }
    };

    check_map(grid_to_distorted_map_A, not excise_A, false);
    check_map(grid_to_distorted_map_B, not excise_B, false);
    check_map(grid_to_inertial_map_A, false, false);
    check_map(grid_to_inertial_map_B, false, false);
    check_map(distorted_to_inertial_map_A, not excise_A,
              (not include_rotation) and (not include_expansion) and excise_A);
    check_map(distorted_to_inertial_map_B, not excise_B,
              (not include_rotation) and (not include_expansion) and excise_B);
  }
}

void check_names() {
  INFO("Check names");
  // These are hard-coded so this is just a regression test
  CHECK(TimeDependentMapOptions::expansion_name == "Expansion"s);
  CHECK(TimeDependentMapOptions::expansion_outer_boundary_name ==
        "ExpansionOuterBoundary"s);
  CHECK(TimeDependentMapOptions::rotation_name == "Rotation"s);
  CHECK(TimeDependentMapOptions::size_names == std::array{"SizeA"s, "SizeB"s});
  CHECK(TimeDependentMapOptions::shape_names ==
        std::array{"ShapeA"s, "ShapeB"s});
}

void test_errors() {
  INFO("Test errors");
  CHECK_THROWS_WITH((TimeDependentMapOptions{1.0, std::nullopt, std::nullopt,
                                             ShapeMapAOptions{1, {}},
                                             ShapeMapBOptions{8, {}}}),
                    Catch::Contains("Initial LMax for object"));
  CHECK_THROWS_WITH((TimeDependentMapOptions{1.0, std::nullopt, std::nullopt,
                                             ShapeMapAOptions{6, {}},
                                             ShapeMapBOptions{0, {}}}),
                    Catch::Contains("Initial LMax for object"));
  CHECK_THROWS_WITH((TimeDependentMapOptions{1.0, std::nullopt, std::nullopt,
                                             std::nullopt, std::nullopt}),
                    Catch::Contains("Time dependent map options were "
                                    "specified, but all options were 'None'."));
  CHECK_THROWS_WITH(
      ([]() {
        TimeDependentMapOptions time_dep_opts{
            1.0, ExpMapOptions{{1.0, 0.0}, 0.0, 0.01}, RotMapOptions{0.0},
            std::nullopt, ShapeMapBOptions{8, {}}};
        time_dep_opts.build_maps(
            std::array{std::array{0.0, 0.0, 0.0}, std::array{0.0, 0.0, 0.0}},
            std::optional<std::pair<double, double>>{std::make_pair(0.1, 1.0)},
            std::nullopt, 100.0);
      }()),
      Catch::Contains("Trying to build the shape map for object " +
                      get_output(domain::ObjectLabel::A)));
  CHECK_THROWS_WITH(
      ([]() {
        TimeDependentMapOptions time_dep_opts{
            1.0, ExpMapOptions{{1.0, 0.0}, 0.0, 0.01}, RotMapOptions{0.0},
            ShapeMapAOptions{8, {}}, std::nullopt};
        time_dep_opts.build_maps(
            std::array{std::array{0.0, 0.0, 0.0}, std::array{0.0, 0.0, 0.0}},
            std::optional<std::pair<double, double>>{std::make_pair(0.1, 1.0)},
            std::optional<std::pair<double, double>>{std::make_pair(0.1, 1.0)},
            100.0);
      }()),
      Catch::Contains("Trying to build the shape map for object " +
                      get_output(domain::ObjectLabel::B)));
  CHECK_THROWS_WITH(
      TimeDependentMapOptions{}.grid_to_distorted_map<domain::ObjectLabel::A>(
          true),
      Catch::Contains("Requesting grid to distorted map with distorted frame "
                      "but shape map options were not specified."));
  CHECK_THROWS_WITH(
      TimeDependentMapOptions{}.grid_to_distorted_map<domain::ObjectLabel::B>(
          true),
      Catch::Contains("Requesting grid to distorted map with distorted frame "
                      "but shape map options were not specified."));
  CHECK_THROWS_WITH(
      TimeDependentMapOptions{}.grid_to_inertial_map<domain::ObjectLabel::A>(
          true),
      Catch::Contains("Requesting grid to inertial map with distorted frame "
                      "but shape map options were not specified."));
  CHECK_THROWS_WITH(
      TimeDependentMapOptions{}.grid_to_inertial_map<domain::ObjectLabel::B>(
          true),
      Catch::Contains("Requesting grid to inertial map with distorted frame "
                      "but shape map options were not specified."));
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      TimeDependentMapOptions{}.has_distorted_frame_options(
          domain::ObjectLabel::None),
      Catch::Contains(
          "object label for TimeDependentMapOptions must be either A or B"));
#endif
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.BinaryCompactObjectHelpers",
                  "[Domain][Unit]") {
  for (const auto& [include_expansion, include_rotation, include_shape_a,
                    include_shape_b] :
       cartesian_product(make_array(true, false), make_array(true, false),
                         make_array(true, false), make_array(true, false))) {
    test(include_expansion, include_rotation, include_shape_a, include_shape_b);
  }
  check_names();
  test_errors();
}
}  // namespace domain::creators::bco
