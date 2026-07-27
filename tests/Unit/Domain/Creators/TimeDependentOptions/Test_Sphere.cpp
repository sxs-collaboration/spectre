// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/Creators/TimeDependentOptions/Sphere.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Framework/TestCreation.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Grid;
struct Distorted;
struct Inertial;
}  // namespace Frame

namespace domain::creators::sphere {
namespace {
// nullopt implies no shape map at all
std::string create_option_string(const std::optional<bool> use_non_zero_shape) {
  return "InitialTime: 1.5\n"
         "ShapeMap:" +
         (use_non_zero_shape.has_value()
              ? "\n"
                "  LMax: 8\n"
                "  CoefficientTruncationLimit: 0.\n"
                "  SizeInitialValues: Auto\n"
                "  InitialValues:" +
                    (use_non_zero_shape.value() ? "\n"
                                                  "    Mass: 1.0\n"
                                                  "    Spin: [0.0, 0.0, 0.0]\n"s
                                                : " Spherical\n"s)
              : " None\n") +
         "RotationMap: None\n"
         "ExpansionMap: None\n"
         "TranslationMap:\n"
         "  InitialValues: [[0.1, -3.2, 1.1], [-0.3, 0.5, -0.7],"
         " [0.1, -0.4, 0.02]]\n"
         "TransitionRotScaleTrans: True\n"
         "NumberOfRadialShellsWithShapeMap: Auto"s;
}
// nullopt implies no shape map at all
void test(const std::optional<bool> use_non_zero_shape) {
  CAPTURE(use_non_zero_shape);
  const double initial_time = 1.5;
  const size_t l_max = 8;

  auto time_dep_options = TestHelpers::test_creation<TimeDependentMapOptions>(
      create_option_string(use_non_zero_shape));

  CHECK(time_dep_options.using_distorted_frame() ==
        use_non_zero_shape.has_value());

  std::unordered_map<std::string, double> expiration_times{
      {TimeDependentMapOptions::shape_name,
       use_non_zero_shape.has_value()
           ? 15.5
           : std::numeric_limits<double>::infinity()},
      {TimeDependentMapOptions::size_name,
       std::numeric_limits<double>::infinity()},
      {TimeDependentMapOptions::translation_name,
       std::numeric_limits<double>::infinity()}};

  // These are hard-coded so this is just a regression test
  CHECK(TimeDependentMapOptions::size_name == "Size"s);
  CHECK(TimeDependentMapOptions::shape_name == "Shape"s);
  CHECK(TimeDependentMapOptions::rotation_name == "Rotation"s);
  CHECK(TimeDependentMapOptions::expansion_name == "Expansion"s);
  CHECK(TimeDependentMapOptions::expansion_outer_boundary_name ==
        "ExpansionOuterBoundary"s);
  CHECK(TimeDependentMapOptions::translation_name == "Translation"s);

  using PP2 = domain::FunctionsOfTime::PiecewisePolynomial<2>;
  using PP3 = domain::FunctionsOfTime::PiecewisePolynomial<3>;
  DataVector size_func{1, 0.0};
  DataVector size_deriv{1, 0.0};
  DataVector size_2nd_deriv{1, 0.0};
  PP3 size{
      initial_time,
      std::array<DataVector, 4>{{size_func, size_deriv, size_2nd_deriv, {0.0}}},
      expiration_times.at(TimeDependentMapOptions::size_name)};
  const DataVector shape_zeros{ylm::Spherepack::spectral_size(l_max, l_max),
                               0.0};
  PP2 shape_all_zero{
      initial_time,
      std::array<DataVector, 3>{shape_zeros, shape_zeros, shape_zeros},
      expiration_times.at(TimeDependentMapOptions::shape_name)};

  DataVector initial_translation_center{{0.1, -3.2, 1.1}};
  DataVector initial_velocity{{-0.3, 0.5, -0.7}};
  DataVector initial_translation_acceleration{{0.1, -0.4, 0.02}};
  PP2 translation_non_zero{
      initial_time,
      std::array<DataVector, 3>{{initial_translation_center, initial_velocity,
                                 initial_translation_acceleration}},
      expiration_times.at(TimeDependentMapOptions::translation_name)};

  const std::array<double, 3> center{-5.0, -0.01, -0.02};
  const double inner_radius = 0.5;
  const std::vector<double> radial_partitions{2.1, 3.1};
  const double outer_radius = 3.9;

  // All of these maps are tested individually. Rather than going through
  // the effort of coming up with a source coordinate and calculating
  // analytically what we would get after it's mapped, we just check that
  // whether it's supposed to be a nullptr and if it's not, if it's the
  // identity and that the jacobians are time dependent.
  const auto check_map = [](const auto& map, const bool is_null,
                            const bool is_identity) {
    if (is_null) {
      CHECK(map == nullptr);
    } else {
      REQUIRE(map != nullptr);
      CHECK(map->is_identity() == is_identity);
      CHECK(map->inv_jacobian_is_time_dependent() == not is_identity);
      CHECK(map->jacobian_is_time_dependent() == not is_identity);
    }
  };

  {
    INFO("Excised");
    time_dep_options.build_maps(center, /* filled */ false, inner_radius,
                                radial_partitions, outer_radius);
    // Inner shell with shape
    check_map(time_dep_options.grid_to_distorted_map(0, 0, false),
              not use_non_zero_shape.has_value(), false);
    check_map(time_dep_options.distorted_to_inertial_map(0, false),
              not use_non_zero_shape.has_value(), false);
    check_map(time_dep_options.grid_to_inertial_map(0, 0, false, false), false,
              false);
    // Shell without shape
    check_map(time_dep_options.grid_to_distorted_map(1, 0, false), true, false);
    check_map(time_dep_options.distorted_to_inertial_map(1, false), true,
              false);
    check_map(time_dep_options.grid_to_inertial_map(1, 0, false, false), false,
              false);
    // Inner S2 shell with shape
    check_map(time_dep_options.grid_to_distorted_map(0, 0, false),
              not use_non_zero_shape.has_value(), false);
    check_map(time_dep_options.distorted_to_inertial_map(0, false),
              not use_non_zero_shape.has_value(), false);
    check_map(time_dep_options.grid_to_inertial_map(0, 0, false, false), false,
              false);
    // S2 Shell without shape
    check_map(time_dep_options.grid_to_distorted_map(1, 0, false), true, false);
    check_map(time_dep_options.distorted_to_inertial_map(1, false), true,
              false);
    check_map(time_dep_options.grid_to_inertial_map(1, 0, false, false), false,
              false);
  }
  {
    INFO("Filled");
    auto filled_time_dep_options =
        TestHelpers::test_creation<TimeDependentMapOptions>(
            create_option_string(use_non_zero_shape));
    filled_time_dep_options.build_maps(center, /* filled */ true, inner_radius,
                                       radial_partitions, outer_radius);
    // Inner shell: cube to sphere
    check_map(filled_time_dep_options.grid_to_distorted_map(0, 0, false),
              not use_non_zero_shape.has_value(), false);
    check_map(filled_time_dep_options.distorted_to_inertial_map(0, false),
              not use_non_zero_shape.has_value(), false);
    check_map(filled_time_dep_options.grid_to_inertial_map(0, 0, false, false),
              false, false);
    // Shape rolloff region
    check_map(filled_time_dep_options.grid_to_distorted_map(1, 0, false),
              not use_non_zero_shape.has_value(), false);
    check_map(filled_time_dep_options.distorted_to_inertial_map(1, false),
              not use_non_zero_shape.has_value(), false);
    check_map(filled_time_dep_options.grid_to_inertial_map(1, 0, false, false),
              false, false);
    // Shell without shape
    check_map(filled_time_dep_options.grid_to_distorted_map(2, 0, false), true,
              false);
    check_map(filled_time_dep_options.distorted_to_inertial_map(2, false), true,
              false);
    check_map(filled_time_dep_options.grid_to_inertial_map(2, 0, false, false),
              false, false);
    // Inner cube
    check_map(filled_time_dep_options.grid_to_distorted_map(2, 0, true), true,
              false);
    check_map(filled_time_dep_options.distorted_to_inertial_map(2, true), true,
              false);
    check_map(filled_time_dep_options.grid_to_inertial_map(2, 0, false, true),
              false, false);
  }

  {
    INFO("Explicit shell count");
    TimeDependentMapOptions time_dep_options_with_two_shape_shells{
        initial_time,
        time_dependent_options::ShapeMapOptions<false,
                                                domain::ObjectLabel::None>{
            l_max, std::nullopt},
        std::nullopt,
        std::nullopt,
        time_dependent_options::TranslationMapOptions<3>{
            std::array{std::array<double, 3>{0.1, -3.2, 1.1},
                       std::array<double, 3>{-0.3, 0.5, -0.7},
                       std::array<double, 3>{0.1, -0.4, 0.02}}},
        true,
        2};
    time_dep_options_with_two_shape_shells.build_maps(
        center, /* filled */ false, inner_radius, radial_partitions,
        outer_radius);
    for (size_t shell = 0; shell < 3; ++shell) {
      const bool has_shape = shell < 2;
      CAPTURE(shell);
      check_map(time_dep_options_with_two_shape_shells.grid_to_distorted_map(
                    shell, 0, false),
                not has_shape, false);
      check_map(
          time_dep_options_with_two_shape_shells.distorted_to_inertial_map(
              shell, false),
          not has_shape, false);
    }
  }

  {
    INFO("Five of eight shells with shape maps");
    const std::vector<double> many_radial_partitions{1.0, 1.5, 2.0, 2.5,
                                                     3.0, 3.5, 3.7};
    TimeDependentMapOptions many_shell_time_dep_options{
        initial_time,
        time_dependent_options::ShapeMapOptions<false,
                                                domain::ObjectLabel::None>{
            l_max, std::nullopt},
        std::nullopt,
        std::nullopt,
        time_dependent_options::TranslationMapOptions<3>{
            std::array{std::array<double, 3>{0.1, -3.2, 1.1},
                       std::array<double, 3>{-0.3, 0.5, -0.7},
                       std::array<double, 3>{0.1, -0.4, 0.02}}},
        true,
        5};
    many_shell_time_dep_options.build_maps(center, /* filled */ false,
                                           inner_radius, many_radial_partitions,
                                           outer_radius);
    for (size_t shell = 0; shell < 8; ++shell) {
      const bool has_shape = shell < 5;
      CAPTURE(shell);
      check_map(
          many_shell_time_dep_options.grid_to_distorted_map(shell, 0, false),
          not has_shape, false);
      check_map(
          many_shell_time_dep_options.distorted_to_inertial_map(shell, false),
          not has_shape, false);
    }
  }

  const auto functions_of_time =
      time_dep_options.create_functions_of_time(expiration_times);

  if (use_non_zero_shape.has_value()) {
    CHECK(
        dynamic_cast<PP3&>(
            *functions_of_time.at(TimeDependentMapOptions::size_name).get()) ==
        size);
    // This will always be zero based on our options above. Just easier to check
    // that way
    CHECK(
        dynamic_cast<PP2&>(
            *functions_of_time.at(TimeDependentMapOptions::shape_name).get()) ==
        shape_all_zero);

    CHECK(dynamic_cast<PP2&>(
              *functions_of_time.at(TimeDependentMapOptions::translation_name)
                   .get()) == translation_non_zero);
  }
}

void test_parse_errors() {
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<TimeDependentMapOptions>(
          "InitialTime: 1.5\n"
          "ShapeMap: None\n"
          "RotationMap: None\n"
          "ExpansionMap: None\n"
          "TranslationMap: None\n"
          "TransitionRotScaleTrans: True\n"),
      Catch::Matchers::ContainsSubstring("NumberOfRadialShellsWithShapeMap"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependentOptions.Sphere",
                  "[Domain][Unit]") {
  test({true});
  test({false});
  test(std::nullopt);
  test_parse_errors();
}
}  // namespace domain::creators::sphere
