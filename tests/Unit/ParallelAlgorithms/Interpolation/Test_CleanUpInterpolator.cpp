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
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Interpolation/Actions/CleanUpInterpolator.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolator.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Interpolation/InterpolatedVars.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox
class DataVector;
template <size_t VolumeDim>
class ElementId;
namespace intrp {
}  // namespace intrp

namespace {

struct OtherId : db::SimpleTag {
  using type = double;
};

template <typename Metavariables>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using simple_tags = typename intrp::Actions::InitializeInterpolator<
      tmpl::list<intrp::Tags::VolumeVarsInfo<Metavariables, OtherId>,
                 intrp::Tags::VolumeVarsInfo<Metavariables, ::Tags::Time>>,
      intrp::Tags::InterpolatedVarsHolders<Metavariables>>::simple_tags;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;
};

struct MockMetavariables {
  struct InterpolationTagA {
    using temporal_id = ::Tags::Time;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
  };
  struct InterpolationTagB {
    using temporal_id = OtherId;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
  };
  struct InterpolationTagC {
    using temporal_id = OtherId;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
  };
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

  double temporal_id = 12.0 / 13.0;

  // Make a VolumeVarsInfo that contains a single temporal_id but
  // no data (since we don't need data for this test).
  std::unordered_map<
      double,
      std::unordered_map<ElementId<3>, intrp::Tags::VolumeVarsInfo<
                                           metavars, OtherId>::Info>>
      volume_vars_info_bc{{temporal_id, {}}};

  std::unordered_map<
      double,
      std::unordered_map<ElementId<3>, intrp::Tags::VolumeVarsInfo<
                                           metavars, ::Tags::Time>::Info>>
      volume_vars_info_a{{temporal_id, {}}};

  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  ActionTesting::emplace_component_and_initialize<interp_component>(
      &runner, 0,
      {0_st,
       typename intrp::Tags::VolumeVarsInfo<metavars, OtherId>::type{
           std::move(volume_vars_info_bc)},
       typename intrp::Tags::VolumeVarsInfo<metavars, ::Tags::Time>::type{
           std::move(volume_vars_info_a)},
       typename intrp::Tags::InterpolatedVarsHolders<metavars>::type{}});
  ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);

  // There should be one temporal_id in VolumeVarsInfo.
  CHECK(
      ActionTesting::get_databox_tag<
          interp_component,
          intrp::Tags::VolumeVarsInfo<metavars, OtherId>>(runner, 0)
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

  // Call the action on InterpolationTagA
  runner.simple_action<
      mock_interpolator<metavars>,
      intrp::Actions::CleanUpInterpolator<metavars::InterpolationTagA>>(
      0, temporal_id);

  // There should still be one temporal_id in VolumeVarsInfo for B and C.
  CHECK(ActionTesting::get_databox_tag<
            interp_component,
            intrp::Tags::VolumeVarsInfo<metavars, ::Tags::Time>>(runner, 0)
            .empty());
  CHECK(
      ActionTesting::get_databox_tag<
          interp_component,
          intrp::Tags::VolumeVarsInfo<metavars, OtherId>>(runner, 0)
          .size() == 1);

  // temporal_ids_when_data_has_been_interpolated should be empty for B and C,
  // and should have already been cleaned up for A.
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
          intrp::Tags::VolumeVarsInfo<metavars, OtherId>>(runner, 0)
          .size() == 1);

  // temporal_ids_when_data_has_been_interpolated should be empty for B,
  // but should contain the correct temporal_id for A and C.
  CHECK(
      get<intrp::Vars::HolderTag<metavars::InterpolationTagA, metavars>>(
          ActionTesting::get_databox_tag<
              interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
              runner, 0))
          .temporal_ids_when_data_has_been_interpolated.empty());
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
  CHECK(ActionTesting::get_databox_tag<
            interp_component,
            intrp::Tags::VolumeVarsInfo<metavars, ::Tags::Time>>(runner, 0)
            .empty());
  CHECK(
      ActionTesting::get_databox_tag<
          interp_component,
          intrp::Tags::VolumeVarsInfo<metavars, OtherId>>(runner, 0)
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
