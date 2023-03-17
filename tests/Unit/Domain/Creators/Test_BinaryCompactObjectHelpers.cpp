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
#include "Utilities/Gsl.hpp"

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

  TimeDependentMapOptions time_dep_options{initial_time, exp_map_options,
                                           angular_velocity, size_A_values,
                                           size_B_values};

  std::unordered_map<std::string, double> expiration_times{
      {TimeDependentMapOptions::expansion_name, 10.0},
      {TimeDependentMapOptions::rotation_name,
       std::numeric_limits<double>::infinity()},
      {gsl::at(TimeDependentMapOptions::size_names, 0), 15.5},
      {gsl::at(TimeDependentMapOptions::size_names, 1),
       std::numeric_limits<double>::infinity()}};

  // These are hard-coded so this is just a regression test
  CHECK(TimeDependentMapOptions::expansion_name == "Expansion"s);
  CHECK(TimeDependentMapOptions::expansion_outer_boundary_name ==
        "ExpansionOuterBoundary"s);
  CHECK(TimeDependentMapOptions::rotation_name == "Rotation"s);
  CHECK(TimeDependentMapOptions::size_names == std::array{"SizeA"s, "SizeB"s});

  using ExpFoT = domain::FunctionsOfTime::PiecewisePolynomial<2>;
  using ExpBdryFoT = domain::FunctionsOfTime::FixedSpeedCubic;
  using RotFoT = domain::FunctionsOfTime::QuaternionFunctionOfTime<3>;
  using SizeFoT = domain::FunctionsOfTime::PiecewisePolynomial<3>;
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
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.BinaryCompactObjectHelpers",
                  "[Domain][Unit]") {
  test();
}
}  // namespace domain::creators::bco
