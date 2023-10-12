// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <iomanip>
#include <memory>
#include <pup.h>
#include <sstream>
#include <string>
#include <unordered_map>

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/OutputTimeBounds.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace {
class TestFoT : public domain::FunctionsOfTime::FunctionOfTime {
 public:
  TestFoT() = default;

// clang-tidy: google-runtime-references
// clang-tidy: cppcoreguidelines-owning-memory,-warnings-as-errors
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(TestFoT);  // NOLINT
#pragma GCC diagnostic pop

  explicit TestFoT(CkMigrateMessage* /*unused*/) {}

  auto get_clone() const -> std::unique_ptr<FunctionOfTime> override {
    return std::make_unique<TestFoT>(*this);
  }

  TestFoT(const double lower_bound, const double upper_bound)
      : lower_bound_(lower_bound), upper_bound_(upper_bound) {}

  std::array<double, 2> time_bounds() const override {
    return {lower_bound_, upper_bound_};
  }

  double expiration_after(const double /*time*/) const override { ERROR(""); }

  std::array<DataVector, 1> func(const double /*t*/) const override {
    ERROR("");
  }
  std::array<DataVector, 2> func_and_deriv(const double /*t*/) const override {
    ERROR("");
  }
  std::array<DataVector, 3> func_and_2_derivs(
      const double /*t*/) const override {
    ERROR("");
  }

  void pup(PUP::er& p) override {
    domain::FunctionsOfTime::FunctionOfTime::pup(p);
    p | lower_bound_;
    p | upper_bound_;
  }

 private:
  double lower_bound_{};
  double upper_bound_{};
};

PUP::able::PUP_ID TestFoT::my_PUP_ID = 0;  // NOLINT

SPECTRE_TEST_CASE("Unit.Domain.FunctionsOfTime.OutputTimeBounds",
                  "[Unit][Domain]") {
  register_classes_with_charm<TestFoT>();
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  functions_of_time["Aang"] = std::make_unique<TestFoT>(0.0, 1.0);
  functions_of_time["Sokka"] = std::make_unique<TestFoT>(2.0, 3.0);
  functions_of_time["Katara"] = std::make_unique<TestFoT>(10.0, 20.0);
  functions_of_time["Toph"] = std::make_unique<TestFoT>(-3.0, 0.0);

  serialize_and_deserialize(functions_of_time);

  const std::string result =
      domain::FunctionsOfTime::ouput_time_bounds(functions_of_time);
  const std::string aang_line =
      " Aang: (0.0000000000000000e+00,1.0000000000000000e+00)\n";
  const std::string sokka_line =
      " Sokka: (2.0000000000000000e+00,3.0000000000000000e+00)\n";
  const std::string katara_line =
      " Katara: (1.0000000000000000e+01,2.0000000000000000e+01)\n";
  const std::string toph_line =
      " Toph: (-3.0000000000000000e+00,0.0000000000000000e+00)\n";
  const std::string title_line = "FunctionsOfTime time bounds:\n";

  // Since it's an unordered map, we can only check that the string contains
  // what we expect
  CHECK(result.find(title_line) != std::string::npos);
  CHECK(result.find(aang_line) != std::string::npos);
  CHECK(result.find(sokka_line) != std::string::npos);
  CHECK(result.find(katara_line) != std::string::npos);
  CHECK(result.find(toph_line) != std::string::npos);
}
}  // namespace
