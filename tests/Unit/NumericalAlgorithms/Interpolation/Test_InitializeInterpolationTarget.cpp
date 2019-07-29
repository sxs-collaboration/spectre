// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <deque>
#include <string>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Domain.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "ParallelBackend/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

/// \cond
class DataVector;
namespace Tags {
template <size_t Dim, typename Frame>
struct Domain;
}  // namespace Tags
/// \endcond

namespace {

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<intrp::Actions::InitializeInterpolationTarget<
          InterpolationTargetTag>>>>;
  using add_options_to_databox =
      typename intrp::Actions::InitializeInterpolationTarget<
          InterpolationTargetTag>::template AddOptionsToDataBox<Metavariables>;
};

struct Metavariables {
  struct InterpolationTargetA {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
  };
  using temporal_id = ::Tags::TimeId;
  using domain_frame = Frame::Inertial;
  static constexpr size_t domain_dim = 3;

  using component_list = tmpl::list<
      mock_interpolation_target<Metavariables, InterpolationTargetA>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialization, Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.InterpolationTarget.Initialize",
                  "[Unit]") {
  using metavars = Metavariables;
  using component =
      mock_interpolation_target<metavars,
                                typename metavars::InterpolationTargetA>;
  const auto domain_creator =
      domain::creators::Shell<Frame::Inertial>(0.9, 4.9, 1, {{5, 5}}, false);

  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  runner.set_phase(Metavariables::Phase::Initialization);
  ActionTesting::emplace_component<component>(&runner, 0,
                                              domain_creator.create_domain());
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  runner.set_phase(Metavariables::Phase::Testing);

  CHECK(ActionTesting::get_databox_tag<
            component, ::intrp::Tags::IndicesOfFilledInterpPoints>(runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<component,
                                       ::intrp::Tags::TemporalIds<metavars>>(
            runner, 0)
            .empty());

  CHECK(ActionTesting::get_databox_tag<component,
                                       ::Tags::Domain<3, Frame::Inertial>>(
            runner, 0) == domain_creator.create_domain());

  const auto test_vars = db::item_type<
      ::Tags::Variables<tmpl::list<gr::Tags::Lapse<DataVector>>>>{};
  CHECK(
      ActionTesting::get_databox_tag<
          component, ::Tags::Variables<typename metavars::InterpolationTargetA::
                                           vars_to_interpolate_to_target>>(
          runner, 0) == test_vars);

  CHECK(::intrp::Tags::IndicesOfFilledInterpPoints::name() ==
        "IndicesOfFilledInterpPoints");
  CHECK(::intrp::Tags::TemporalIds<metavars>::name() == "TemporalIds");
}

}  // namespace
