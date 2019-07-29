// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>

#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolatedVars.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "ParallelBackend/AddOptionsToDataBox.hpp"
#include "ParallelBackend/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <boost/variant/get.hpp>

/// \cond
class DataVector;
/// \endcond

namespace {

template <typename Metavariables>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<intrp::Actions::InitializeInterpolator>>>;
};

struct Metavariables {
  struct InterpolatorTargetA {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
  };
  using temporal_id = ::Tags::TimeId;
  static constexpr size_t domain_dim = 3;
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags = tmpl::list<InterpolatorTargetA>;

  using component_list = tmpl::list<mock_interpolator<Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialization, Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.Initialize",
                  "[Unit]") {
  using metavars = Metavariables;
  using component = mock_interpolator<metavars>;
  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  runner.set_phase(Metavariables::Phase::Initialization);
  ActionTesting::emplace_component<component>(&runner, 0);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  runner.set_phase(Metavariables::Phase::Testing);

  CHECK(ActionTesting::get_databox_tag<component,
                                       ::intrp::Tags::NumberOfElements>(
            runner, 0) == 0);
  CHECK(ActionTesting::get_databox_tag<component,
                                       ::intrp::Tags::VolumeVarsInfo<metavars>>(
            runner, 0)
            .empty());

  const auto& holders = ActionTesting::get_databox_tag<
      component, ::intrp::Tags::InterpolatedVarsHolders<metavars>>(runner, 0);
  // Check that 'holders' has a tag corresponding to
  // 'metavars::InterpolatorTargetA'
  const auto& holder =
      get<intrp::Vars::HolderTag<metavars::InterpolatorTargetA, metavars>>(
          holders);
  CHECK(holder.infos.empty());
  // Check that 'holders' has only one tag.
  CHECK(holders.size() == 1);

  // check tag names
  CHECK(::intrp::Tags::VolumeVarsInfo<metavars>::name() == "VolumeVarsInfo");
  CHECK(::intrp::Tags::InterpolatedVarsHolders<metavars>::name() ==
        "InterpolatedVarsHolders");
  CHECK(::intrp::Tags::NumberOfElements::name() == "NumberOfElements");
}
}  // namespace
