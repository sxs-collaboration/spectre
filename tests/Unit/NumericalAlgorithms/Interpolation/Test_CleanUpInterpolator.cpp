// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <pup.h>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Interpolation/CleanUpInterpolator.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolatedVars.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox
class DataVector;
template <size_t VolumeDim>
class ElementId;
namespace intrp {
}  // namespace intrp
/// \endcond

namespace {

template <typename Metavariables>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using simple_tags =
      db::get_items<typename intrp::Actions::InitializeInterpolator<
          intrp::Tags::VolumeVarsInfo<Metavariables>,
          intrp::Tags::InterpolatedVarsHolders<Metavariables>>::
                        return_tag_list>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;
};

struct MockMetavariables {
  struct InterpolationTagA {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
  };
  struct InterpolationTagB {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
  };
  struct InterpolationTagC {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
  };
  using temporal_id = ::Tags::TimeStepId;
  static constexpr size_t volume_dim = 3;
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags =
      tmpl::list<InterpolationTagA, InterpolationTagB, InterpolationTagC>;

  using component_list = tmpl::list<mock_interpolator<MockMetavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.CleanUp", "[Unit]") {
  using metavars = MockMetavariables;
  using interp_component = mock_interpolator<metavars>;

  Slab slab(0.0, 1.0);
  TimeStepId temporal_id(true, 0, Time(slab, Rational(12, 13)));

  // Make a VolumeVarsInfo that contains a single temporal_id but
  // no data (since we don't need data for this test).
  std::unordered_map<
      typename metavars::temporal_id::type,
      std::unordered_map<ElementId<3>,
                         intrp::Tags::VolumeVarsInfo<metavars>::Info>>
      volume_vars_info{{temporal_id, {}}};

  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  ActionTesting::emplace_component_and_initialize<interp_component>(
      &runner, 0,
      {0_st,
       typename intrp::Tags::VolumeVarsInfo<metavars>::type{
           std::move(volume_vars_info)},
       typename intrp::Tags::InterpolatedVarsHolders<metavars>::type{}});
  ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);

  // There should be one temporal_id in VolumeVarsInfo.
  CHECK(ActionTesting::get_databox_tag<interp_component,
                                       intrp::Tags::VolumeVarsInfo<metavars>>(
            runner, 0)
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

  // Call the action on InterpolationTagA
  runner.simple_action<
      mock_interpolator<metavars>,
      intrp::Actions::CleanUpInterpolator<metavars::InterpolationTagA>>(
      0, temporal_id);

  // There should still be one temporal_id in VolumeVarsInfo.
  CHECK(ActionTesting::get_databox_tag<interp_component,
                                       intrp::Tags::VolumeVarsInfo<metavars>>(
            runner, 0)
            .size() == 1);

  // temporal_ids_when_data_has_been_interpolated should be empty for B and C,
  // but should contain the correct temporal_id for A.
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagA, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.size() == 1);
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagA, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.count(temporal_id) ==
      1);
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

  // Call the action on InterpolationTagC
  runner.simple_action<
      mock_interpolator<metavars>,
      intrp::Actions::CleanUpInterpolator<metavars::InterpolationTagC>>(
      0, temporal_id);

  // There should still be one temporal_id in VolumeVarsInfo.
  CHECK(ActionTesting::get_databox_tag<interp_component,
                                       intrp::Tags::VolumeVarsInfo<metavars>>(
            runner, 0)
            .size() == 1);

  // temporal_ids_when_data_has_been_interpolated should be empty for B,
  // but should contain the correct temporal_id for A and C.
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagA, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.size() == 1);
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagA, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.count(temporal_id) ==
      1);
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagC, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.size() == 1);
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagC, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.count(temporal_id) ==
      1);
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagB, metavars>>(
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
  CHECK(ActionTesting::get_databox_tag<interp_component,
                                       intrp::Tags::VolumeVarsInfo<metavars>>(
            runner, 0)
            .empty());

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

  // There should be no queued actions; verify this.
  CHECK(runner.is_simple_action_queue_empty<mock_interpolator<metavars>>(0));
}

}  // namespace
