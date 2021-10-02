// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <memory>
#include <string>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionsOfTimeAreReady.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/MockDistributedObject.hpp"
#include "Framework/MockRuntimeSystem.hpp"
#include "Framework/MockRuntimeSystemFreeFunctions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using FunctionMap = domain::Tags::FunctionsOfTime::type;

struct OtherFunctionsOfTime : db::SimpleTag {
  using type = FunctionMap;
};

struct UpdateFoT {
  static void apply(const gsl::not_null<FunctionMap*> functions,
                    const std::string& name, const double expiration) {
    (*functions).at(name)->reset_expiration_time(expiration);
  }
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;

  using initialization_tags = tmpl::list<Tags::Time>;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTime, OtherFunctionsOfTime>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Testing,
      tmpl::list<domain::Actions::CheckFunctionsOfTimeAreReady>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.FunctionsOfTimeAreReady", "[Domain][Unit]") {
  Parallel::register_classes_with_charm<
      domain::FunctionsOfTime::PiecewisePolynomial<2>>();
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using component = Component<Metavariables>;
  const component* const component_p = nullptr;

  const std::array<DataVector, 3> fot_init{{{0.0}, {0.0}, {0.0}}};
  FunctionMap functions_of_time{};
  functions_of_time["FunctionA"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          0.0, fot_init, 1.0);
  functions_of_time["FunctionB"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          0.0, fot_init, 1.0);

  FunctionMap other_functions_of_time{};
  other_functions_of_time["OtherA"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          0.0, fot_init, 0.0);
  other_functions_of_time["OtherB"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          0.0, fot_init, 0.0);

  MockRuntimeSystem runner{
      {}, {std::move(functions_of_time), std::move(other_functions_of_time)}};
  ActionTesting::emplace_array_component<component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, 0, 2.0);
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);

  auto& cache = ActionTesting::cache<component>(runner, 0);

  // Test the free function.  For this section of the test, the
  // "standard" tags domain::Tags::FunctionsOfTime is ready, but this
  // code should never examine it.
  {
    // Neither function ready
    CHECK(not domain::functions_of_time_are_ready<OtherFunctionsOfTime>(
        cache, 0, component_p, 0.5));
    CHECK(not domain::functions_of_time_are_ready<OtherFunctionsOfTime>(
        cache, 0, component_p, 0.5, std::array{"OtherA"s}));

    // Make OtherA ready
    Parallel::mutate<OtherFunctionsOfTime, UpdateFoT>(cache, "OtherA"s, 123.0);

    CHECK(domain::functions_of_time_are_ready<OtherFunctionsOfTime>(
        cache, 0, component_p, 0.5, std::array{"OtherA"s}));
    CHECK(not domain::functions_of_time_are_ready<OtherFunctionsOfTime>(
        cache, 0, component_p, 0.5, std::array{"OtherA"s, "OtherB"s}));

    // Make OtherB ready
    Parallel::mutate<OtherFunctionsOfTime, UpdateFoT>(cache, "OtherB"s, 456.0);

    CHECK(domain::functions_of_time_are_ready<OtherFunctionsOfTime>(
        cache, 0, component_p, 0.5, std::array{"OtherA"s, "OtherB"s}));
    CHECK(domain::functions_of_time_are_ready<OtherFunctionsOfTime>(
        cache, 0, component_p, 0.5));
  }

  // Test the action.  This should automatically look at
  // domain::Tags::FunctionsOfTime.
  {
    // Neither function ready
    CHECK(not ActionTesting::next_action_if_ready<component>(
        make_not_null(&runner), 0));

    // Make OtherA ready
    Parallel::mutate<domain::Tags::FunctionsOfTime, UpdateFoT>(
        cache, "FunctionA"s, 5.0);

    CHECK(not ActionTesting::next_action_if_ready<component>(
        make_not_null(&runner), 0));

    // Make OtherB ready
    Parallel::mutate<domain::Tags::FunctionsOfTime, UpdateFoT>(
        cache, "FunctionB"s, 10.0);

    CHECK(ActionTesting::next_action_if_ready<component>(make_not_null(&runner),
                                                         0));
  }
}
