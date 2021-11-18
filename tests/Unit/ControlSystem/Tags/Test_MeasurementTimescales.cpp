// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/OptionTags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"

namespace {
const double initial_time = 2.0;

struct Metavariables {
  static constexpr size_t volume_dim = 1;
};

class TestCreator : public DomainCreator<1> {
 public:
  explicit TestCreator(const bool add_controlled)
      : add_controlled_(add_controlled) {}
  Domain<1> create_domain() const override { ERROR(""); }
  std::vector<std::array<size_t, 1>> initial_extents() const override {
    ERROR("");
  }
  std::vector<std::array<size_t, 1>> initial_refinement_levels()
      const override {
    ERROR("");
  }
  auto functions_of_time() const -> std::unordered_map<
      std::string,
      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override {
    const std::array<DataVector, 3> initial_values{{{-1.0}, {-2.0}, {-3.0}}};

    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        result{};
    if (add_controlled_) {
      result.insert(
          {"Controlled1",
           std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
               initial_time, initial_values, initial_time + 7.0)});
      result.insert(
          {"Controlled2",
           std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
               initial_time, initial_values, initial_time + 10.0)});
      result.insert(
          {"Controlled3",
           std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
               initial_time, initial_values, initial_time + 0.5)});
    }
    result.insert(
        {"Uncontrolled",
         std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
             initial_time, initial_values,
             std::numeric_limits<double>::infinity())});
    return result;
  }

 private:
  bool add_controlled_{};
};

void test_measurement_tag() {
  INFO("Test measurement tag");
  using measurement_tag = control_system::Tags::MeasurementTimescales;
  static_assert(
      tmpl::size<measurement_tag::option_tags<Metavariables>>::value == 2);
  using Creator =
      tmpl::front<measurement_tag::option_tags<Metavariables>>::type;
  const double time_step = 0.2;
  {
    const Creator creator = std::make_unique<TestCreator>(true);

    const measurement_tag::type timescales =
        measurement_tag::create_from_options<Metavariables>(creator, time_step);
    CHECK(timescales.size() == 3);
    // The lack of expiration is a placeholder until the control systems
    // have been implemented sufficiently to manage their timescales.
    CHECK(timescales.at("Controlled1")->time_bounds() ==
          std::array{initial_time, std::numeric_limits<double>::infinity()});
    CHECK(timescales.at("Controlled1")->func(2.0)[0] == DataVector{time_step});
    CHECK(timescales.at("Controlled1")->func(3.0)[0] == DataVector{time_step});
    CHECK(timescales.at("Controlled2")->time_bounds() ==
          std::array{initial_time, std::numeric_limits<double>::infinity()});
    CHECK(timescales.at("Controlled2")->func(2.0)[0] == DataVector{time_step});
    CHECK(timescales.at("Controlled2")->func(3.0)[0] == DataVector{time_step});
    CHECK(timescales.at("Controlled3")->time_bounds() ==
          std::array{initial_time, std::numeric_limits<double>::infinity()});
    CHECK(timescales.at("Controlled3")->func(2.5)[0] == DataVector{time_step});
  }
  {
    const Creator creator = std::make_unique<TestCreator>(false);

    // Verify that negative time steps are accepted with no control
    // systems.
    const measurement_tag::type timescales =
        measurement_tag::create_from_options<Metavariables>(creator,
                                                            -time_step);
    CHECK(timescales.empty());
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.Tags.MeasurementTimescales",
                  "[ControlSystem][Unit]") {
  test_measurement_tag();
}

// [[OutputRegex, Control systems can only be used in forward-in-time
// evolutions.]]
SPECTRE_TEST_CASE("Unit.ControlSystem.Tags.MeasurementTimescales.Backwards",
                  "[ControlSystem][Unit]") {
  ERROR_TEST();
  using measurement_tag = control_system::Tags::MeasurementTimescales;
  const std::unique_ptr<DomainCreator<1>> creator =
      std::make_unique<TestCreator>(true);
  measurement_tag::create_from_options<Metavariables>(creator, -1.0);
}
