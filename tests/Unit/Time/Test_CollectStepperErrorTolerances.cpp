// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>

#include "Time/CollectStepperErrorTolerances.hpp"
#include "Time/RequestsStepperErrorTolerances.hpp"
#include "Time/StepperErrorTolerances.hpp"

namespace {
class Tolerances final : public RequestsStepperErrorTolerances {
 public:
  explicit Tolerances(
      std::unordered_map<std::type_index, StepperErrorTolerances> tolerances)
      : tolerances_(std::move(tolerances)) {}

  std::unordered_map<std::type_index, StepperErrorTolerances> tolerances()
      const override {
    return tolerances_;
  }

 private:
  std::unordered_map<std::type_index, StepperErrorTolerances> tolerances_{};
};

class NotTolerances {
 public:
  virtual ~NotTolerances() = default;

  static std::unordered_map<std::type_index, StepperErrorTolerances>
  tolerances() {
    // Should never be called.
    CHECK(false);
    return {};
  }
};

SPECTRE_TEST_CASE("Unit.Time.Test_CollectStepperErrorTolerances",
                  "[Unit][Time]") {
  const StepperErrorTolerances tols_32{
      .estimates = StepperErrorTolerances::Estimates::StepperOrder,
      .absolute = 1e-3,
      .relative = 1e-2};
  const StepperErrorTolerances tols_54{
      .estimates = StepperErrorTolerances::Estimates::StepperOrder,
      .absolute = 1e-5,
      .relative = 1e-4};

  {
    std::unordered_map<std::type_index, StepperErrorTolerances> tolerances{};
    collect_stepper_error_tolerances(&tolerances, NotTolerances{});
    CHECK(tolerances.empty());
  }

  {
    std::unordered_map<std::type_index, StepperErrorTolerances> tolerances{
        {typeid(int), tols_32}};
    collect_stepper_error_tolerances(&tolerances, NotTolerances{});
    CHECK(tolerances.size() == 1);
    CHECK(tolerances.at(typeid(int)) == tols_32);
  }

  {
    std::unordered_map<std::type_index, StepperErrorTolerances> tolerances{};
    collect_stepper_error_tolerances(&tolerances, Tolerances{{}});
    CHECK(tolerances.empty());
  }

  {
    std::unordered_map<std::type_index, StepperErrorTolerances> tolerances{};
    collect_stepper_error_tolerances(&tolerances,
                                     Tolerances{{{typeid(int), tols_32}}});
    CHECK(tolerances.size() == 1);
    CHECK(tolerances.at(typeid(int)) == tols_32);
  }

  {
    std::unordered_map<std::type_index, StepperErrorTolerances> tolerances{
        {typeid(int), tols_32}};
    collect_stepper_error_tolerances(&tolerances,
                                     Tolerances{{{typeid(int), tols_32}}});
    CHECK(tolerances.size() == 1);
    CHECK(tolerances.at(typeid(int)) == tols_32);
  }

  {
    std::unordered_map<std::type_index, StepperErrorTolerances> tolerances{
        {typeid(int), tols_32}};
    collect_stepper_error_tolerances(&tolerances,
                                     Tolerances{{{typeid(double), tols_54}}});
    CHECK(tolerances.size() == 2);
    CHECK(tolerances.at(typeid(int)) == tols_32);
    CHECK(tolerances.at(typeid(double)) == tols_54);
  }

  {
    std::unordered_map<std::type_index, StepperErrorTolerances> tolerances{};
    collect_stepper_error_tolerances(
        &tolerances, Tolerances{{{typeid(int), StepperErrorTolerances{}}}});
    CHECK(tolerances.empty());
  }

  {
    std::unordered_map<std::type_index, StepperErrorTolerances> tolerances{
        {typeid(int), tols_32}};
    collect_stepper_error_tolerances(
        &tolerances, Tolerances{{{typeid(int), StepperErrorTolerances{}}}});
    CHECK(tolerances.size() == 1);
    CHECK(tolerances.at(typeid(int)) == tols_32);
  }

  {
    std::unordered_map<std::type_index, StepperErrorTolerances> tolerances{
        {typeid(int), tols_32}};
    CHECK_THROWS_WITH(
        collect_stepper_error_tolerances(&tolerances,
                                         Tolerances{{{typeid(int), tols_54}}}),
        Catch::Matchers::ContainsSubstring(
            "All time stepping error tolerances for one set of variables must "
            "be the same"));
  }
}
}  // namespace
