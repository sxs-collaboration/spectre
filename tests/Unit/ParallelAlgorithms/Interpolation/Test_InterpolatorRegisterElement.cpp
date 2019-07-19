// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Interpolation/InitializeInterpolator.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Interpolation/InterpolatedVars.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Interpolation/InterpolatorRegisterElement.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <boost/variant/get.hpp>

/// \cond
class DataVector;
namespace intrp {
namespace Tags {
struct NumberOfElements;
}  // namespace Tags
}  // namespace intrp
/// \endcond

namespace {

template <typename Metavariables>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<intrp::Actions::InitializeInterpolator>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Registration, tmpl::list<>>>;
};

struct MockMetavariables {
  struct InterpolatorTargetA {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
  };
  using temporal_id = ::Tags::TimeId;
  static constexpr size_t domain_dim = 3;
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags = tmpl::list<InterpolatorTargetA>;

  using component_list = tmpl::list<mock_interpolator<MockMetavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialization, Registration, Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.RegisterElement",
                  "[Unit]") {
  using metavars = MockMetavariables;
  using component = mock_interpolator<metavars>;
  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  runner.set_phase(metavars::Phase::Initialization);
  ActionTesting::emplace_component<component>(&runner, 0);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  runner.set_phase(metavars::Phase::Registration);

  CHECK(ActionTesting::get_databox_tag<component,
                                       ::intrp::Tags::NumberOfElements>(
            runner, 0) == 0);

  runner.simple_action<component, ::intrp::Actions::RegisterElement>(0);

  CHECK(ActionTesting::get_databox_tag<component,
                                       ::intrp::Tags::NumberOfElements>(
            runner, 0) == 1);

  runner.simple_action<component, ::intrp::Actions::RegisterElement>(0);

  CHECK(ActionTesting::get_databox_tag<component,
                                       ::intrp::Tags::NumberOfElements>(
            runner, 0) == 2);
}

}  // namespace
