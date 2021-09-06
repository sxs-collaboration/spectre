// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ControlSystem/Tags.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/OptionTags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Literals.hpp"
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
  Domain<1> create_domain() const noexcept override { ERROR(""); }
  std::vector<std::array<size_t, 1>> initial_extents() const noexcept override {
    ERROR("");
  }
  std::vector<std::array<size_t, 1>> initial_refinement_levels()
      const noexcept override {
    ERROR("");
  }
  auto functions_of_time() const noexcept -> std::unordered_map<
      std::string,
      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override {
    const std::array<DataVector, 3> initial_values{{{-1.0}, {-2.0}, {-3.0}}};

    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        result{};
    if (add_controlled_) {
      result.insert(
          {"Controlled",
           std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
               initial_time, initial_values, 7.0)});
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
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.Tags", "[ControlSystem][Unit]") {
  using tag = control_system::Tags::MeasurementTimescales;
  TestHelpers::db::test_simple_tag<tag>("MeasurementTimescales");

  static_assert(tmpl::size<tag::option_tags<Metavariables>>::value == 2);
  using Creator = tmpl::front<tag::option_tags<Metavariables>>::type;
  const double time_step = 0.2;
  {
    const Creator creator = std::make_unique<TestCreator>(true);

    const tag::type timescales =
        tag::create_from_options<Metavariables>(creator, time_step);
    CHECK(timescales.size() == 1);
    // The lack of expiration is a placeholder until the control systems
    // have been implemented sufficiently to manage their timescales.
    CHECK(timescales.at("Controlled")->time_bounds() ==
          std::array{initial_time, std::numeric_limits<double>::infinity()});
    CHECK(timescales.at("Controlled")->func(2.0)[0] == DataVector{time_step});
    CHECK(timescales.at("Controlled")->func(3.0)[0] == DataVector{time_step});
  }
  {
    const Creator creator = std::make_unique<TestCreator>(false);

    // Verify that negative time steps are accepted with no control
    // systems.
    const tag::type timescales =
        tag::create_from_options<Metavariables>(creator, -time_step);
    CHECK(timescales.empty());
  }
}

// [[OutputRegex, Control systems can only be used in forward-in-time
// evolutions.]]
SPECTRE_TEST_CASE("Unit.ControlSystem.Tags.Backwards",
                  "[ControlSystem][Unit]") {
  ERROR_TEST();
  using tag = control_system::Tags::MeasurementTimescales;
  const std::unique_ptr<DomainCreator<1>> creator =
      std::make_unique<TestCreator>(true);
  tag::create_from_options<Metavariables>(creator, -1.0);
}
