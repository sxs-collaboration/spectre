// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <functional>
#include <pup.h>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Interpolation/Actions/CleanUpInterpolator.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolator.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Interpolation/InterpolatedVars.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox
class DataVector;
template <size_t VolumeDim>
class ElementId;
namespace intrp {}  // namespace intrp

namespace {

template <typename Metavariables>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using simple_tags = typename intrp::Actions::InitializeInterpolator<
      tmpl::list<intrp::Tags::VolumeVarsInfo<Metavariables, ::Tags::TimeStepId>,
                 intrp::Tags::VolumeVarsInfo<Metavariables, ::Tags::Time>>,
      intrp::Tags::InterpolatedVarsHolders<Metavariables>>::simple_tags;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
};

template <typename Metavariables>
struct mock_element_array {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using simple_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
};

struct MockMetavariables {
  struct InterpolationTagA {
    using temporal_id = ::Tags::Time;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
  };
  struct InterpolationTagB {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
  };
  struct InterpolationTagC {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
  };
  struct InterpolationTagD {
    using temporal_id = ::Tags::Time;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    // Doesn't have to be a real component, this type alias just needs to exist
    // and not be the interpolator
    template <typename Metavariables>
    using interpolating_component = mock_element_array<Metavariables>;
  };
  static constexpr size_t volume_dim = 3;
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags =
      tmpl::list<InterpolationTagA, InterpolationTagB, InterpolationTagC,
                 InterpolationTagD>;

  using component_list = tmpl::list<mock_interpolator<MockMetavariables>,
                                    mock_element_array<MockMetavariables>>;
};

template <typename interp_component, typename InterpolationTargetTag,
          typename Metavariables, typename TemporalId>
bool temporal_ids_when_data_has_been_interpolated_contains(
    const ActionTesting::MockRuntimeSystem<Metavariables>& runner,
    const TemporalId& temporal_id) {
  const auto& finished_temporal_ids =
      get<intrp::Vars::HolderTag<InterpolationTargetTag, Metavariables>>(
          ActionTesting::get_databox_tag<
              interp_component,
              intrp::Tags::InterpolatedVarsHolders<Metavariables>>(runner, 0))
          .temporal_ids_when_data_has_been_interpolated;
  return alg::found(finished_temporal_ids, temporal_id);
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.CleanUp", "[Unit]") {
  using metavars = MockMetavariables;
  using interp_component = mock_interpolator<metavars>;

  Slab slab(0.0, 1.0);
  TimeStepId temporal_id(true, 0, Time(slab, Rational(12, 13)));

  // Make a VolumeVarsInfo that contains a single temporal_id but
  // no data (since we don't need data for this test).
  std::unordered_map<
      TimeStepId,
      std::unordered_map<ElementId<3>, intrp::Tags::VolumeVarsInfo<
                                           metavars, ::Tags::TimeStepId>::Info>>
      volume_vars_info_bc{{temporal_id, {}}};

  std::unordered_map<
      double,
      std::unordered_map<ElementId<3>, intrp::Tags::VolumeVarsInfo<
                                           metavars, ::Tags::Time>::Info>>
      volume_vars_info_ad{{temporal_id.substep_time(), {}}};

  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  ActionTesting::emplace_component_and_initialize<interp_component>(
      &runner, 0,
      {0_st,
       typename intrp::Tags::VolumeVarsInfo<metavars, ::Tags::TimeStepId>::type{
           std::move(volume_vars_info_bc)},
       typename intrp::Tags::VolumeVarsInfo<metavars, ::Tags::Time>::type{
           std::move(volume_vars_info_ad)},
       typename intrp::Tags::InterpolatedVarsHolders<metavars>::type{}});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  // There should be one temporal_id in VolumeVarsInfo.
  CHECK(
      ActionTesting::get_databox_tag<
          interp_component,
          intrp::Tags::VolumeVarsInfo<metavars, ::Tags::TimeStepId>>(runner, 0)
          .size() == 1);
  CHECK(ActionTesting::get_databox_tag<
            interp_component,
            intrp::Tags::VolumeVarsInfo<metavars, ::Tags::Time>>(runner, 0)
            .size() == 1);

  // temporal_ids_when_data_has_been_interpolated should be empty for each tag.
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagA, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.empty());
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagB, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.empty());
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagC, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.empty());
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagD, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.empty());

  // Call the action on InterpolationTagA
  runner.simple_action<
      mock_interpolator<metavars>,
      intrp::Actions::CleanUpInterpolator<metavars::InterpolationTagA>>(
      0, temporal_id.substep_time());

  // There should still be one temporal_id in VolumeVarsInfo for B and C.
  CHECK(ActionTesting::get_databox_tag<
            interp_component,
            intrp::Tags::VolumeVarsInfo<metavars, ::Tags::Time>>(runner, 0)
            .empty());
  CHECK(
      ActionTesting::get_databox_tag<
          interp_component,
          intrp::Tags::VolumeVarsInfo<metavars, ::Tags::TimeStepId>>(runner, 0)
          .size() == 1);

  // temporal_ids_when_data_has_been_interpolated should be empty for B, C, and
  // D (because D isn't using the interpolator),
  // but should have a single entry for A with the correct value.
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagA, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.size() == 1);
  CHECK(temporal_ids_when_data_has_been_interpolated_contains<
        interp_component, metavars::InterpolationTagA>(
      runner, temporal_id.substep_time()));
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagB, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.empty());
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagC, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.empty());
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagD, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.empty());

  // Call the action on InterpolationTagC
  runner.simple_action<
      mock_interpolator<metavars>,
      intrp::Actions::CleanUpInterpolator<metavars::InterpolationTagC>>(
      0, temporal_id);

  // There should still be one temporal_id in VolumeVarsInfo for C.
  CHECK(ActionTesting::get_databox_tag<
            interp_component,
            intrp::Tags::VolumeVarsInfo<metavars, ::Tags::Time>>(runner, 0)
            .empty());
  CHECK(
      ActionTesting::get_databox_tag<
          interp_component,
          intrp::Tags::VolumeVarsInfo<metavars, ::Tags::TimeStepId>>(runner, 0)
          .size() == 1);

  // temporal_ids_when_data_has_been_interpolated should be empty for B and D,
  // but should contain the correct temporal_id for A and C.
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagA, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.size() == 1);
  CHECK(temporal_ids_when_data_has_been_interpolated_contains<
        interp_component, metavars::InterpolationTagA>(
      runner, temporal_id.substep_time()));
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagC, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.size() == 1);
  CHECK(temporal_ids_when_data_has_been_interpolated_contains<
        interp_component, metavars::InterpolationTagC>(runner, temporal_id));
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagB, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.empty());
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagD, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.empty());

  // Call the action on InterpolationTagB. This will clean up everything
  // since all the tags have now cleaned up.
  runner.simple_action<
      mock_interpolator<metavars>,
      intrp::Actions::CleanUpInterpolator<metavars::InterpolationTagB>>(
      0, temporal_id);

  // There should be no temporal_ids in VolumeVarsInfo.
  CHECK(ActionTesting::get_databox_tag<
            interp_component,
            intrp::Tags::VolumeVarsInfo<metavars, ::Tags::Time>>(runner, 0)
            .empty());
  CHECK(
      ActionTesting::get_databox_tag<
          interp_component,
          intrp::Tags::VolumeVarsInfo<metavars, ::Tags::TimeStepId>>(runner, 0)
          .empty());

  // temporal_ids_when_data_has_been_interpolated should contain the correct
  // values for each tag.  One entry per tag except for target D.
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagA, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.size() == 1);
  CHECK(temporal_ids_when_data_has_been_interpolated_contains<
        interp_component, metavars::InterpolationTagA>(
      runner, temporal_id.substep_time()));
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagB, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.size() == 1);
  CHECK(temporal_ids_when_data_has_been_interpolated_contains<
        interp_component, metavars::InterpolationTagB>(runner, temporal_id));
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagC, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.size() == 1);
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagD, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.empty());
  CHECK(temporal_ids_when_data_has_been_interpolated_contains<
        interp_component, metavars::InterpolationTagC>(runner, temporal_id));

  // There should be no queued actions; verify this.
  CHECK(runner.is_simple_action_queue_empty<mock_interpolator<metavars>>(0));

  // Now ensure that cleaning up the interpolator more than 1000 times
  // does not result in more than 1000 entries in
  // temporal_ids_when_data_has_been_interpolated.  (The number 1000
  // is hardcoded in CleanUpInterpolator.hpp and must agree with the next
  // line)
  constexpr size_t finished_temporal_ids_max_size = 1000;
  constexpr size_t number_of_test_calls = finished_temporal_ids_max_size + 10;
  for (size_t i = 0; i < number_of_test_calls; ++i) {
    // In normal usage, different calls to CleanUpInterpolator are
    // made for different temporal_ids, but the test should work fine
    // if the same temporal_id is duplicated multiple times.
    runner.simple_action<
        mock_interpolator<metavars>,
        intrp::Actions::CleanUpInterpolator<metavars::InterpolationTagA>>(
        0, temporal_id.substep_time());
  }
  // There should be exactly 1000 entries, even though it was called
  // more than 1000 times.
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagA, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.size() ==
      finished_temporal_ids_max_size);
}

}  // namespace
