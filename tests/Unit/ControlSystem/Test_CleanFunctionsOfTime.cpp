// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "ControlSystem/CleanFunctionsOfTime.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename Metavariables>
struct Component {
  using chare_type = ActionTesting::MockNodeGroupChare;
  using array_index = size_t;
  using metavariables = Metavariables;
  using const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTime,
                 control_system::Tags::MeasurementTimescales>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

struct Metavars {
  using component_list = tmpl::list<Component<Metavars>>;
};

SPECTRE_TEST_CASE("Unit.ControlSystem.CleanFunctionsOfTime",
                  "[Unit][ControlSystem]") {
  // ActionTesting doesn't support reductions, so this just tests the
  // reduction action as a simple action.

  domain::FunctionsOfTime::register_derived_with_charm();
  using component = Component<Metavars>;

  const std::initializer_list<std::string> names{"A", "B", "C"};

  domain::FunctionsOfTime::PiecewisePolynomial<0> fot_template(
      0.0, std::array{DataVector{1.0}}, 1.7);
  fot_template.update(1.7, DataVector{1.0}, 2.9);
  fot_template.update(2.9, DataVector{1.0}, 5.8);
  fot_template.update(5.8, DataVector{1.0}, 7.2);
  fot_template.update(7.2, DataVector{1.0}, 10.1);

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      init_functions_of_time{};
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      init_measurement_timescales{};

  for (const auto& name : names) {
    init_functions_of_time[name] = fot_template.get_clone();
    init_measurement_timescales[name] = fot_template.get_clone();
  }

  ActionTesting::MockRuntimeSystem<Metavars> runner{
      {},
      {std::move(init_functions_of_time),
       std::move(init_measurement_timescales)}};
  ActionTesting::emplace_nodegroup_component<component>(make_not_null(&runner));

  ActionTesting::simple_action<component,
                               control_system::CleanFunctionsOfTimeAction>(
      make_not_null(&runner), 0_st, 5.0);

  const auto& cache = ActionTesting::cache<component>(runner, 0_st);
  const auto& functions_of_time =
      Parallel::get<domain::Tags::FunctionsOfTime>(cache);
  const auto& measurement_timescales =
      Parallel::get<control_system::Tags::MeasurementTimescales>(cache);

  for (const auto& name : names) {
    CHECK(functions_of_time.at(name)->time_bounds()[0] > 1.0);
    CHECK(functions_of_time.at(name)->time_bounds()[0] <= 5.0);
    CHECK(measurement_timescales.at(name)->time_bounds()[0] > 1.0);
    CHECK(measurement_timescales.at(name)->time_bounds()[0] <= 5.0);
  }
}
}  // namespace
