// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>

#include "Domain/CoordinateMaps/TimeDependent/Rotation.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Evolution/Triggers/NumberOfOrbits.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"

#include <cmath>

namespace {
struct Metavariables {
  using component_list = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Trigger, tmpl::list<Triggers::NumberOfOrbits>>>;
  };
};

using RotationMap = domain::CoordinateMaps::TimeDependent::Rotation<3>;

void test() {
  const std::string f_of_t_name = "Rotation";
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  DataVector axis{{0.0, 1.0, 1.0}};
  axis /= sqrt(2.0);
  std::array<DataVector, 4> init_func = {axis, axis * 0.6, axis * 0.0,
                                         axis * 0.0};
  const std::array<DataVector, 1> init_quat{DataVector{
      {cos(0.5), axis[0] * sin(0.5), axis[1] * sin(0.5), axis[2] * sin(0.5)}}};
  domain::FunctionsOfTime::QuaternionFunctionOfTime<3> quat_f_of_t{
      0.0, init_quat, init_func, 20.0};
  functions_of_time[f_of_t_name] =
      std::make_unique<domain::FunctionsOfTime::QuaternionFunctionOfTime<3>>(
          quat_f_of_t);
  const std::array<double, 3> number_of_orbits{0.5, 1.0, 2.0};
  for (size_t i = 0; i < 3; i++) {
    double time = 0.0;
    Triggers::NumberOfOrbits trigger{number_of_orbits[i]};
    while (time <= 20.0) {
      bool is_triggered = trigger(time, functions_of_time);
      bool expected_is_triggered =
          (quat_f_of_t.full_angle(time) / (2.0 * M_PI) >=
           number_of_orbits[i]);
      CHECK(is_triggered == expected_is_triggered);
      time += 1.0;
    }
  }

  TestHelpers::test_creation<std::unique_ptr<Trigger>, Metavariables>(
      "NumberOfOrbits:\n"
      "  Value: 3.0");
}

void test_errors() {
  const std::string f_of_t_name = "NeonPegasus";
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  DataVector axis{{0.0, 0.0, 1.0}};
  std::array<DataVector, 3> init_func = {axis, axis, axis};
  const std::array<DataVector, 1> init_quat{
      DataVector{{0.0, axis[0], axis[1], axis[2]}}};
  functions_of_time[f_of_t_name] =
      std::make_unique<domain::FunctionsOfTime::QuaternionFunctionOfTime<2>>(
          0.0, init_quat, init_func, 20.0);

  Triggers::NumberOfOrbits trigger{0.1};

  CHECK_THROWS_WITH(trigger(0.0, functions_of_time),
                    Catch::Matchers::ContainsSubstring(
                        "NumberOfOrbits trigger can only be used when the "
                        "rotation map is active"));
}

SPECTRE_TEST_CASE("Unit.Evolution.Triggers.NumberOfOrbits",
                  "[Unit][Evolution]") {
  test();
  test_errors();
}
}  // namespace
