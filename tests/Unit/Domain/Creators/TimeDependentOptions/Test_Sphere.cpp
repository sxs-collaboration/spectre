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
         " [0.1, -0.4, 0.02]]\n"s;
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
  const double outer_radius = 2.1;
  const double transition_inner_radius = 3.1;
  const double transition_outer_radius = 3.9;

  const auto functions_of_time =
      time_dep_options.create_functions_of_time(inner_radius, expiration_times);

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

  for (const bool include_distorted : make_array(true, false)) {
    for (const bool use_rigid : make_array(true, false)) {
      time_dep_options.build_maps(
          center, std::pair<double, double>{inner_radius, outer_radius},
          std::pair<double, double>{transition_inner_radius,
                                    transition_outer_radius});

      if ((not use_non_zero_shape.has_value()) and include_distorted) {
        CHECK_THROWS_WITH(
            time_dep_options.grid_to_distorted_map(include_distorted),
            Catch::Matchers::ContainsSubstring(
                "Requesting grid to distorted map with distorted frame but "
                "shape map options were not specified."));
        CHECK_THROWS_WITH(
            time_dep_options.grid_to_inertial_map(include_distorted, use_rigid),
            Catch::Matchers::ContainsSubstring(
                "Requesting grid to inertial map with distorted frame but "
                "shape map options were not specified."));
        continue;
      }

      const auto grid_to_distorted_map =
          time_dep_options.grid_to_distorted_map(include_distorted);
      const auto grid_to_inertial_map =
          time_dep_options.grid_to_inertial_map(include_distorted, use_rigid);
      const auto distorted_to_inertial_map =
          time_dep_options.distorted_to_inertial_map(include_distorted);

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
          CHECK(map->is_identity() == is_identity);
          CHECK(map->inv_jacobian_is_time_dependent() == not is_identity);
          CHECK(map->jacobian_is_time_dependent() == not is_identity);
        }
      };

      // There is no null pointer in the grid to inertial map
      check_map(grid_to_inertial_map, false, false);

      check_map(grid_to_distorted_map, not include_distorted, false);

      // If no shape distortion, there is only the rotation, expansion and
      // translation maps
      check_map(distorted_to_inertial_map, not include_distorted, false);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependentOptions.Sphere",
                  "[Domain][Unit]") {
  test({true});
  test({false});
  test(std::nullopt);
}
}  // namespace domain::creators::sphere
