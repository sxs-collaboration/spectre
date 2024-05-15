// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/IntegratedFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace domain {
namespace {
void test(const gsl::not_null<FunctionsOfTime::FunctionOfTime*> f_of_t,
          const double dt, const DataVector& positions,
          const DataVector& velocities, const bool rotation) {
  auto* f_of_t_derived =
      dynamic_cast<FunctionsOfTime::IntegratedFunctionOfTime*>(f_of_t.get());
  const FunctionsOfTime::IntegratedFunctionOfTime f_of_t_derived_copy =
      *f_of_t_derived;
  CHECK(*f_of_t_derived == f_of_t_derived_copy);
  for (size_t i = 0; i < positions.size() - 1; ++i) {
    const double t = static_cast<double>(i) * dt;
    const auto pos_and_vel = f_of_t->func_and_deriv(t);
    const auto pos = f_of_t->func(t);

    if (rotation) {
      const DataVector quaternion{cos(positions.at(i) / 2.), 0., 0.,
                                  sin(positions.at(i) / 2.)};
      const DataVector dt_quaternion{
          -sin(positions.at(i) / 2.) * velocities.at(i) / 2., 0., 0.,
          cos(positions.at(i) / 2.) * velocities.at(i) / 2.};
      CHECK_ITERABLE_APPROX(pos_and_vel[0], quaternion);
      CHECK_ITERABLE_APPROX(pos_and_vel[1], dt_quaternion);
      CHECK_ITERABLE_APPROX(pos[0], quaternion);
    } else {
      CHECK(approx(pos_and_vel[0][0]) == positions.at(i));
      CHECK(approx(pos_and_vel[1][0]) == velocities.at(i));
      CHECK(approx(pos[0][0]) == positions.at(i));
    }
    const auto time_bounds = f_of_t->time_bounds();
    CHECK(time_bounds[0] == approx(std::max(0., t - 99.5 * dt)));
    CHECK(time_bounds[1] == approx(t + 0.5 * dt));

    f_of_t_derived->update(time_bounds[1],
                           DataVector{positions[i + 1], velocities[i + 1]},
                           time_bounds[1] + dt);
    CHECK(f_of_t->expiration_after(t) == approx(t + 0.5 * dt));
    CHECK(*f_of_t_derived != f_of_t_derived_copy);
  }
}

void test_out_of_order_update() {
  const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> func(
      std::make_unique<domain::FunctionsOfTime::IntegratedFunctionOfTime>(
          0.0, std::array<double, 2>{0., 0.5}, 0.5, false));

  func->update(2.5, {{2.5, 3.}}, 3.0);
  func->update(1.5, {{1.5, 2.}}, 2.0);

  CHECK_THROWS_WITH(
      func->func(0.7),
      Catch::Matchers::ContainsSubstring("Attempt to evaluate at time") and
          Catch::Matchers::ContainsSubstring(
              ", which is after the expiration time 5"));
  CHECK_THROWS_WITH(
      func->func(1.3),
      Catch::Matchers::ContainsSubstring("Attempt to evaluate at time 1.30") and
          Catch::Matchers::ContainsSubstring(
              ", which is after the expiration time 5"));
  CHECK_THROWS_WITH(
      func->func(1.9),
      Catch::Matchers::ContainsSubstring("Attempt to evaluate at time") and
          Catch::Matchers::ContainsSubstring(
              ", which is after the expiration time 5"));

  func->update(0.5, {{0.5, 1.0}}, 1.0);

  // This one should pass now
  CHECK(func->func_and_deriv(0.7) ==
        std::array<DataVector, 2>{DataVector{0.5}, DataVector{1.}});
  CHECK_THROWS_WITH(
      func->func(1.3),
      Catch::Matchers::ContainsSubstring("Attempt to evaluate at time 1.30") and
          Catch::Matchers::ContainsSubstring(
              ", which is after the expiration time 1.0"));
  CHECK_THROWS_WITH(
      func->func(1.9),
      Catch::Matchers::ContainsSubstring("Attempt to evaluate at time") and
          Catch::Matchers::ContainsSubstring(
              ", which is after the expiration time 1.0"));

  func->update(2.0, {{2., 2.5}}, 2.5);

  CHECK_THROWS_WITH(
      func->func(1.3),
      Catch::Matchers::ContainsSubstring("Attempt to evaluate at time 1.30") and
          Catch::Matchers::ContainsSubstring(
              ", which is after the expiration time 1.0"));
  CHECK_THROWS_WITH(
      func->func(1.9),
      Catch::Matchers::ContainsSubstring("Attempt to evaluate at time") and
          Catch::Matchers::ContainsSubstring(
              ", which is after the expiration time 1.0"));

  func->update(1.0, {{1., 1.5}}, 1.5);

  CHECK(func->func_and_deriv(1.3) ==
        std::array<DataVector, 2>{DataVector{1.}, DataVector{1.5}});
  CHECK(func->func_and_deriv(1.9) ==
        std::array<DataVector, 2>{DataVector{1.5}, DataVector{2.}});
  CHECK(func->func_and_deriv(2.2) ==
        std::array<DataVector, 2>{DataVector{2.}, DataVector{2.5}});
  CHECK(func->func_and_deriv(2.8) ==
        std::array<DataVector, 2>{DataVector{2.5}, DataVector{3.}});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.FunctionsOfTime.IntegratedFunctionOfTime",
                  "[Domain][Unit]") {
  FunctionsOfTime::register_derived_with_charm();
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist(-10., 10.);
  const size_t used_for_size = 300;
  const auto random_positions = make_with_random_values<DataVector>(
      make_not_null(&gen), make_not_null(&dist), used_for_size);
  const auto random_velocities = make_with_random_values<DataVector>(
      make_not_null(&gen), make_not_null(&dist), used_for_size);
  double t = 0.0;
  const double dt = 0.6;
  const std::array<double, 2> init_func{random_positions[0],
                                        random_velocities[0]};
  {
    INFO("Test no rotation with simple construction.");
    FunctionsOfTime::IntegratedFunctionOfTime f_of_t(t, init_func, t + 0.5 * dt,
                                                     false);
    FunctionsOfTime::IntegratedFunctionOfTime f_of_t2 =
        serialize_and_deserialize(f_of_t);
    test(make_not_null(&f_of_t), dt, random_positions, random_velocities,
         false);
    test(make_not_null(&f_of_t2), dt, random_positions, random_velocities,
         false);
  }
  {
    INFO("Test rotation with simple construction.");
    FunctionsOfTime::IntegratedFunctionOfTime f_of_t(t, init_func, t + 0.5 * dt,
                                                     true);
    FunctionsOfTime::IntegratedFunctionOfTime f_of_t2 =
        serialize_and_deserialize(f_of_t);
    test(make_not_null(&f_of_t), dt, random_positions, random_velocities, true);
    test(make_not_null(&f_of_t2), dt, random_positions, random_velocities,
         true);
  }
  {
    INFO("Test no rotation with base class construction.");
    std::unique_ptr<FunctionsOfTime::FunctionOfTime> f_of_t1 =
        std::make_unique<FunctionsOfTime::IntegratedFunctionOfTime>(
            t, init_func, t + 0.5 * dt, false);
    std::unique_ptr<FunctionsOfTime::FunctionOfTime> f_of_t2 =
        serialize_and_deserialize(f_of_t1);
    std::unique_ptr<FunctionsOfTime::FunctionOfTime> f_of_t3 =
        f_of_t1->get_clone();
    test(make_not_null(f_of_t1.get()), dt, random_positions, random_velocities,
         false);
    test(make_not_null(f_of_t2.get()), dt, random_positions, random_velocities,
         false);
    test(make_not_null(f_of_t3.get()), dt, random_positions, random_velocities,
         false);
  }
  {
    INFO("Test rotation with base class construction.");
    std::unique_ptr<FunctionsOfTime::FunctionOfTime> f_of_t1 =
        std::make_unique<FunctionsOfTime::IntegratedFunctionOfTime>(
            t, init_func, t + 0.5 * dt, true);
    std::unique_ptr<FunctionsOfTime::FunctionOfTime> f_of_t2 =
        serialize_and_deserialize(f_of_t1);
    std::unique_ptr<FunctionsOfTime::FunctionOfTime> f_of_t3 =
        f_of_t1->get_clone();
    test(make_not_null(f_of_t1.get()), dt, random_positions, random_velocities,
         true);
    test(make_not_null(f_of_t2.get()), dt, random_positions, random_velocities,
         true);
    test(make_not_null(f_of_t3.get()), dt, random_positions, random_velocities,
         true);
  }
  test_out_of_order_update();
}
}  // namespace domain
