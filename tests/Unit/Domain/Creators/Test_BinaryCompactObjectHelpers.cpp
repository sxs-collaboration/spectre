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
namespace {
void test() {
  const std::array<double, 2> exp_values{1.0, 0.0};
  const double exp_outer_boundary_velocity = -0.01;
  const double exp_outer_boundary_timescale = 25.0;
  ExpMapOptions exp_map_options{exp_values, exp_outer_boundary_velocity,
                                exp_outer_boundary_timescale};

  const double initial_time = 1.5;
  const std::array<double, 3> angular_velocity{0.2, -0.4, 0.6};
  const std::array<double, 3> size_A_values{0.9, 0.08, 0.007};
  const std::array<double, 3> size_B_values{-0.001, -0.02, -0.3};
  const size_t l_max_A = 8;
  const size_t l_max_B = 10;

  TimeDependentMapOptions time_dep_options{
      initial_time,  exp_map_options, angular_velocity, size_A_values,
      size_B_values, l_max_A,         l_max_B};

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

  // These are hard-coded so this is just a regression test
  CHECK(TimeDependentMapOptions::expansion_name == "Expansion"s);
  CHECK(TimeDependentMapOptions::expansion_outer_boundary_name ==
        "ExpansionOuterBoundary"s);
  CHECK(TimeDependentMapOptions::rotation_name == "Rotation"s);
  CHECK(TimeDependentMapOptions::size_names == std::array{"SizeA"s, "SizeB"s});
  CHECK(TimeDependentMapOptions::shape_names ==
        std::array{"ShapeA"s, "ShapeB"s});

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

  CHECK(dynamic_cast<ExpFoT&>(
            *functions_of_time.at(TimeDependentMapOptions::expansion_name)
                 .get()) == expansion);
  CHECK(dynamic_cast<ExpBdryFoT&>(
            *functions_of_time
                 .at(TimeDependentMapOptions::expansion_outer_boundary_name)
                 .get()) == expansion_outer_boundary);
  CHECK(dynamic_cast<RotFoT&>(
            *functions_of_time.at(TimeDependentMapOptions::rotation_name)
                 .get()) == rotation);
  CHECK(
      dynamic_cast<SizeFoT&>(
          *functions_of_time.at(gsl::at(TimeDependentMapOptions::size_names, 0))
               .get()) == size_A);
  CHECK(
      dynamic_cast<SizeFoT&>(
          *functions_of_time.at(gsl::at(TimeDependentMapOptions::size_names, 1))
               .get()) == size_B);
  CHECK(dynamic_cast<ShapeFoT&>(
            *functions_of_time
                 .at(gsl::at(TimeDependentMapOptions::shape_names, 0))
                 .get()) == shape_A);
  CHECK(dynamic_cast<ShapeFoT&>(
            *functions_of_time
                 .at(gsl::at(TimeDependentMapOptions::shape_names, 1))
                 .get()) == shape_B);

  const std::array<std::array<double, 3>, 2> centers{
      std::array{5.0, 0.01, 0.02}, std::array{-5.0, -0.01, -0.02}};
  const double domain_outer_radius = 20.0;

  for (const auto& [excise_A, excise_B] :
       cartesian_product(make_array(true, false), make_array(true, false))) {
    std::optional<double> inner_radius_A{};
    std::optional<double> inner_radius_B{};
    std::optional<double> outer_radius_A{};
    std::optional<double> outer_radius_B{};
    if (excise_A) {
      inner_radius_A = 0.8;
      outer_radius_A = 3.2;
    }
    if (excise_B) {
      inner_radius_B = 0.5;
      outer_radius_B = 2.1;
    }

    const std::array<std::optional<double>, 2> inner_radii{inner_radius_A,
                                                           inner_radius_B};
    const std::array<std::optional<double>, 2> outer_radii{outer_radius_A,
                                                           outer_radius_B};

    time_dep_options.build_maps(centers, inner_radii, outer_radii,
                                domain_outer_radius);

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
    const auto check_map = [](const auto& map, const bool is_null) {
      if (is_null) {
        CHECK(map == nullptr);
      } else {
        CHECK_FALSE(map->is_identity());
        CHECK(map->inv_jacobian_is_time_dependent());
        CHECK(map->jacobian_is_time_dependent());
      }
    };

    check_map(grid_to_distorted_map_A, not excise_A);
    check_map(grid_to_distorted_map_B, not excise_B);
    check_map(grid_to_inertial_map_A, false);
    check_map(grid_to_inertial_map_B, false);
    check_map(distorted_to_inertial_map_A, not excise_A);
    check_map(distorted_to_inertial_map_B, not excise_B);
  }
}

void test_errors() {
  CHECK_THROWS_WITH((TimeDependentMapOptions{1.0, {}, {}, {}, {}, 1, 8}),
                    Catch::Contains("Initial LMax for object"));
  CHECK_THROWS_WITH((TimeDependentMapOptions{1.0, {}, {}, {}, {}, 6, 0}),
                    Catch::Contains("Initial LMax for object"));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.BinaryCompactObjectHelpers",
                  "[Domain][Unit]") {
  test();
  test_errors();
}
}  // namespace domain::creators::bco
