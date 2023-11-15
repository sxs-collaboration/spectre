// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/IdPair.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/BlockId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Framework/ActionTesting.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorReceivePoints.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorReceiveVolumeData.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorRegisterElement.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/TryToInterpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolatedVars.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/LineSegment.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace intrp::Actions {
template <typename InterpolationTargetTag>
struct InterpolationTargetReceiveVars;
}  // namespace intrp::Actions
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace intrp::Tags {
template <typename Metavariables>
struct InterpolatedVarsHolders;
template <typename Metavariables>
struct TemporalIds;
}  // namespace intrp::Tags
namespace Tags {
template <typename TagsList>
struct Variables;
}  // namespace Tags

namespace {
enum WhichElement { Left, Right };

struct PointsInWhichElement : db::SimpleTag {
  using type = WhichElement;
};

// So there will be a point in every element in the y direction
constexpr size_t number_of_points = 20;
// Target points along line x=0.5, z=0.0 from y=0.0 to y=1.0
constexpr std::array<double, 3> begin{0.5, 0.0, 0.0};
constexpr std::array<double, 3> end{0.5, 1.0, 0.0};

template <typename InterpolationTargetTag, typename Frame>
struct SequentialLineSegment
    : tt::ConformsTo<intrp::protocols::ComputeTargetPoints> {
  using const_global_cache_tags = tmpl::list<>;
  using is_sequential = std::true_type;
  using frame = Frame;
  using simple_tags = tmpl::list<PointsInWhichElement>;

  template <typename Metavariables, typename DbTags>
  static tnsr::I<DataVector, 3, Frame> points(
      const db::DataBox<DbTags>& box,
      const tmpl::type_<Metavariables>& /*meta*/) {
    const auto& which_element = get<PointsInWhichElement>(box);

    const double fractional_distance = 1.0 / (number_of_points - 1);
    tnsr::I<DataVector, 3, Frame> target_points(number_of_points);
    for (size_t n = 0; n < number_of_points; ++n) {
      for (size_t d = 0; d < 3; ++d) {
        target_points.get(d)[n] =
            gsl::at(begin, d) + static_cast<double>(n) * fractional_distance *
                                    (gsl::at(end, d) - gsl::at(begin, d));

        // Move the points slightly into the left/right element depending on the
        // options
        if (d == 0) {
          if (which_element == WhichElement::Left) {
            target_points.get(d)[n] -= 0.1;
          } else if (which_element == WhichElement::Right) {
            target_points.get(d)[n] += 0.1;
          }
        }
      }
    }
    return target_points;
  }

  template <typename Metavariables, typename DbTags, typename TemporalId>
  static tnsr::I<DataVector, 3, Frame> points(
      const db::DataBox<DbTags>& box, const tmpl::type_<Metavariables>& meta,
      const TemporalId& /*temporal_id*/) {
    return points(box, meta);
  }
};

size_t num_calls_of_target_receive_vars = 0;
template <typename InterpolationTargetTag>
struct MockInterpolationTargetReceiveVars {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex, typename TemporalId>
  static void apply(
      db::DataBox<DbTags>& /*box*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const std::vector<::Variables<
          typename InterpolationTargetTag::vars_to_interpolate_to_target>>&
      /*vars_src*/,
      const std::vector<std::vector<size_t>>& /*global_offsets*/,
      const TemporalId& /*temporal_id*/) {
    // InterpolationTargetReceiveVars will not be called in this test,
    // because we are not supplying volume data (so try_to_interpolate
    // inside TryToInterpolate.hpp will not actually interpolate). However, the
    // compiler thinks that InterpolationTargetReceiveVars might be called, so
    // we mock it so that everything compiles.
    //
    // Note that try_to_interpolate and InterpolationTargetReceiveVars have
    // already been tested by other tests.

    // Here we increment a variable so that later we can
    // verify that this wasn't called.
    ++num_calls_of_target_receive_vars;
  }
};

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  static_assert(
      tt::assert_conforms_to_v<InterpolationTargetTag,
                               intrp::protocols::InterpolationTargetTag>);
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using component_being_mocked =
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>;
  using const_global_cache_tags =
      tmpl::list<domain::Tags::Domain<Metavariables::volume_dim>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<tmpl::list<>>,
                     intrp::Actions::InitializeInterpolationTarget<
                         Metavariables, InterpolationTargetTag>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;

  using replace_these_simple_actions =
      tmpl::list<intrp::Actions::InterpolationTargetReceiveVars<
          typename Metavariables::InterpolationTargetA>>;
  using with_these_simple_actions =
      tmpl::list<MockInterpolationTargetReceiveVars<
          typename Metavariables::InterpolationTargetA>>;
};

template <typename Metavariables>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<::intrp::Actions::InitializeInterpolator<
              intrp::Tags::VolumeVarsInfo<Metavariables, ::Tags::TimeStepId>,
              intrp::Tags::InterpolatedVarsHolders<Metavariables>>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
  using component_being_mocked = void;  // not needed.
};

struct Metavariables {
  struct InterpolationTargetA
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points =
        SequentialLineSegment<InterpolationTargetA, Frame::Inertial>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::ObserveTimeSeriesOnSurface<
            tmpl::list<>, InterpolationTargetA>>;
  };
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;
  static constexpr size_t volume_dim = 3;
  using component_list =
      tmpl::list<mock_interpolation_target<Metavariables, InterpolationTargetA>,
                 mock_interpolator<Metavariables>>;
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.ReceivePoints",
                  "[Unit]") {
  domain::creators::register_derived_with_charm();
  using metavars = Metavariables;
  using target_tag = typename metavars::InterpolationTargetA;
  using compute_points = typename target_tag::compute_target_points;
  using target_component = mock_interpolation_target<metavars, target_tag>;
  using interp_component = mock_interpolator<metavars>;
  // Eight elements
  const auto domain_creator = domain::creators::Brick{
      std::array{0.0, 0.0, 0.0}, std::array{1.0, 1.0, 1.0},
      std::array{1_st, 2_st, 0_st}, std::array{2_st, 2_st, 2_st},
      std::array{false, false, false}};
  const Domain<3> domain = domain_creator.create_domain();

  ActionTesting::MockRuntimeSystem<metavars> runner{
      {domain_creator.create_domain(), ::Verbosity::Silent},
      {},
      std::vector<std::size_t>{2_st}};
  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  // Two components of the interpolator
  for (size_t array_index = 0; array_index < 2; array_index++) {
    ActionTesting::emplace_component<interp_component>(&runner, array_index);
    for (size_t i = 0; i < 2; ++i) {
      ActionTesting::next_action<interp_component>(make_not_null(&runner),
                                                   array_index);
    }
  }
  ActionTesting::emplace_array_component_and_initialize<target_component>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0_st,
      {});
  auto& box =
      ActionTesting::get_databox<target_component>(make_not_null(&runner), 0);
  // Set the points to be in the x<0.5 elements first
  db::mutate<PointsInWhichElement>(
      [](const gsl::not_null<WhichElement*> which_element) {
        *which_element = WhichElement::Left;
      },
      make_not_null(&box));
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  // Make eight element ids and meshes
  std::vector<ElementId<3>> element_ids{};
  element_ids.reserve(8);
  std::vector<Mesh<3>> meshes{};
  meshes.reserve(8);
  for (const auto& block : domain.blocks()) {
    const auto initial_ref_levs =
        domain_creator.initial_refinement_levels()[block.id()];
    auto elem_ids = initial_element_ids(block.id(), initial_ref_levs);
    element_ids.insert(element_ids.end(), elem_ids.begin(), elem_ids.end());
  }
  for (const auto& element_id : element_ids) {
    meshes.emplace_back(domain_creator.initial_extents()[element_id.block_id()],
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto);
  }
  // Same number of grid points in each element so just create one lapse..
  Variables<tmpl::list<gr::Tags::Lapse<DataVector>>> lapse{
      meshes[0].number_of_grid_points(), 0.0};

  // Make sure that we have four elements on each core registered,
  for (size_t array_index = 0; array_index < 2; array_index++) {
    for (size_t num_elements = 0; num_elements < 4; num_elements++) {
      runner.simple_action<interp_component, ::intrp::Actions::RegisterElement>(
          array_index);
    }
  }

  const auto create_block_logical_coords = [&domain, &box]() {
    const tnsr::I<DataVector, 3, Frame::Inertial> points =
        compute_points::points(box, tmpl::type_<metavars>{});
    return block_logical_coordinates(domain, points);
  };

  auto block_logical_coords = create_block_logical_coords();
  Slab slab(0.0, 1.0);
  TimeStepId temporal_id(true, 0, Time(slab, Rational(11, 15)));

  const auto& get_holders = [&runner](const size_t array_index) {
    return ActionTesting::get_databox_tag<
        interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(
        runner, array_index);
  };
  const auto& get_holder = [&get_holders](const size_t array_index) {
    const auto& holders = get_holders(array_index);
    return get<
        intrp::Vars::HolderTag<metavars::InterpolationTargetA, metavars>>(
        holders);
  };

  // Send points to both interpolator cores. Not much should happen. We send at
  // iteration 1 so we can test receiving points before, at, and after this
  // iteration.
  for (size_t array_index = 0; array_index < 2; array_index++) {
    runner.simple_action<interp_component, intrp::Actions::ReceivePoints<
                                               metavars::InterpolationTargetA>>(
        array_index, temporal_id, block_logical_coords, 1_st);
    const auto& holder = get_holder(array_index);

    // Should now be a single info in holder, indexed by temporal_id.
    CHECK(holder.infos.size() == 1);
    const auto& info = holder.infos.at(temporal_id);
    // We haven't done any interpolation because we haven't received
    // volume data from any Elements, so these fields should be empty.
    CHECK(holder.temporal_ids_when_data_has_been_interpolated.empty());
    CHECK(info.interpolation_is_done_for_these_elements.empty());
    CHECK(info.global_offsets.empty());
    CHECK(info.vars.empty());
    // But block_coord_holders should be filled and the iteration should be 1
    CHECK(info.block_coord_holders == block_logical_coords);
    CHECK(info.iteration == 1_st);
    // There should be no more queued actions; verify this.
    CHECK(runner.is_simple_action_queue_empty<interp_component>(array_index));
    // Make sure that the action was not called.
    CHECK(num_calls_of_target_receive_vars == 0);
  }

  using vars_info_tag =
      intrp::Tags::VolumeVarsInfo<metavars, ::Tags::TimeStepId>;
  const auto& get_vars_info = [&runner](const size_t array_index) {
    return ActionTesting::get_databox_tag<interp_component, vars_info_tag>(
        runner, array_index);
  };

  // Send data to first interpolator core from the first three elements
  for (size_t i = 0; i < 3; i++) {
    INFO("Element " + get_output(i));
    runner.simple_action<
        interp_component,
        intrp::Actions::InterpolatorReceiveVolumeData<Tags::TimeStepId>>(
        0_st, temporal_id, element_ids[i], meshes[i], lapse);

    // Ensure the volume data got to the interpolator
    const auto& vars_info = get_vars_info(0_st);
    CHECK(vars_info.count(temporal_id) == 1);
    CHECK(vars_info.at(temporal_id).count(element_ids[i]) == 1);

    // We should have interpolated some data but not called the
    // target_receive_vars
    const auto& holder = get_holder(0_st);
    CHECK(holder.temporal_ids_when_data_has_been_interpolated.empty());
    CHECK(holder.infos.size() == 1);
    const auto& info = holder.infos.at(temporal_id);
    CHECK(info.interpolation_is_done_for_these_elements.count(element_ids[i]) ==
          1);
    CHECK(info.block_coord_holders == block_logical_coords);
    CHECK(info.iteration == 1_st);
    CHECK(info.global_offsets.size() == i + 1);
    CHECK(info.vars.size() == i + 1);

    // There should be no queued actions and no calls to target_receive_vars
    CHECK(runner.is_simple_action_queue_empty<target_component>(0_st));
    CHECK(num_calls_of_target_receive_vars == 0);
  }

  // Send data to first interpolator core from the fourth element
  runner.simple_action<
      interp_component,
      intrp::Actions::InterpolatorReceiveVolumeData<Tags::TimeStepId>>(
      0_st, temporal_id, element_ids[3], meshes[3], lapse);

  {
    INFO("Element 3");
    // Ensure the volume data got to the interpolator
    const auto& vars_info = get_vars_info(0_st);
    CHECK(vars_info.count(temporal_id) == 1);
    CHECK(vars_info.at(temporal_id).count(element_ids[3]) == 1);

    // We should have interpolated some data and called the target_receive_vars.
    // This should have cleaned things up
    const auto& holder = get_holder(0_st);
    CHECK(holder.temporal_ids_when_data_has_been_interpolated.empty());
    CHECK(holder.infos.empty());

    // There should be one queued action; verify this, but not called yet
    CHECK(runner.number_of_queued_simple_actions<target_component>(0_st) == 1);
    CHECK(num_calls_of_target_receive_vars == 0);
    // Invoke the action and check that it's called
    ActionTesting::invoke_queued_simple_action<target_component>(
        make_not_null(&runner), 0_st);
    CHECK(num_calls_of_target_receive_vars == 1);
  }

  // Send data to second interpolator core from fifth and sixth elements
  for (size_t i = 4; i < 6; i++) {
    INFO("Element " + get_output(i));
    runner.simple_action<
        interp_component,
        intrp::Actions::InterpolatorReceiveVolumeData<Tags::TimeStepId>>(
        1_st, temporal_id, element_ids[i], meshes[i], lapse);

    // Ensure the volume data got to the interpolator
    const auto& vars_info = get_vars_info(1_st);
    CHECK(vars_info.count(temporal_id) == 1);
    CHECK(vars_info.at(temporal_id).count(element_ids[i]) == 1);

    // We should have a holder for this temporal id, but there shouldn't be any
    // interpolated data. The block logical coords should be the new ones
    const auto& holder = get_holder(1_st);
    CHECK(holder.temporal_ids_when_data_has_been_interpolated.empty());
    CHECK(holder.infos.size() == 1);
    const auto& info = holder.infos.at(temporal_id);
    CHECK(info.interpolation_is_done_for_these_elements.count(element_ids[i]) ==
          1);
    CHECK(info.block_coord_holders == block_logical_coords);
    CHECK(info.iteration == 1_st);
    CHECK(info.global_offsets.empty());
    CHECK(info.vars.empty());

    // There should be no queued actions and no extra calls to
    // target_receive_vars
    CHECK(runner.is_simple_action_queue_empty<target_component>(0_st));
    CHECK(num_calls_of_target_receive_vars == 1);
  }

  // Now send new points to the interpolator for x>0.5
  db::mutate<PointsInWhichElement>(
      [](const gsl::not_null<WhichElement*> which_element) {
        *which_element = WhichElement::Right;
      },
      make_not_null(&box));
  block_logical_coords = create_block_logical_coords();

  // First send with iteration 0. These points should be ignored
  for (size_t array_index = 0; array_index < 2; array_index++) {
    runner.simple_action<interp_component, intrp::Actions::ReceivePoints<
                                               metavars::InterpolationTargetA>>(
        array_index, temporal_id, block_logical_coords, 0_st);
  }

  // Send data to second interpolator core from the seventh element. We should
  // not have done an interpolation to these new points because of iteration 1
  runner.simple_action<
      interp_component,
      intrp::Actions::InterpolatorReceiveVolumeData<Tags::TimeStepId>>(
      1_st, temporal_id, element_ids[6], meshes[6], lapse);

  {
    INFO("Element 6 no interpolate");
    // Ensure the volume data got to the interpolator
    const auto& vars_info = get_vars_info(1_st);
    CHECK(vars_info.count(temporal_id) == 1);
    CHECK(vars_info.at(temporal_id).count(element_ids[6]) == 1);

    // Now we should NOT have interpolated data on the 3 elements that have
    // received volume data thus far.
    const auto& holder = get_holder(1_st);
    CHECK(holder.temporal_ids_when_data_has_been_interpolated.empty());
    CHECK(holder.infos.size() == 1);
    const auto& info = holder.infos.at(temporal_id);
    CHECK(info.interpolation_is_done_for_these_elements.count(element_ids[6]) ==
          1);
    // These should still be the old coords
    CHECK_FALSE(info.block_coord_holders == block_logical_coords);
    CHECK(info.iteration == 1_st);
    CHECK(info.global_offsets.empty());
    CHECK(info.vars.empty());

    // There should be no queued actions and no extra calls to
    // target_receive_vars
    CHECK(runner.is_simple_action_queue_empty<target_component>(0_st));
    CHECK(num_calls_of_target_receive_vars == 1);
  }

  // Now send with iteration 1. One core 0, this shouldn't error because we have
  // cleaned up points. But on core 1 this should error
  runner.simple_action<interp_component, intrp::Actions::ReceivePoints<
                                             metavars::InterpolationTargetA>>(
      0_st, temporal_id, block_logical_coords, 1_st);
  CHECK_THROWS_WITH(
      (runner.simple_action<
          interp_component,
          intrp::Actions::ReceivePoints<metavars::InterpolationTargetA>>(
          1_st, temporal_id, block_logical_coords, 1_st)),
      Catch::Matchers::ContainsSubstring(
          "Interpolator received target points at iteration 1 twice."));

  // Now send with iteration 2. These points should overwrite the previous ones
  // and an interpolation should happen
  for (size_t array_index = 0; array_index < 2; array_index++) {
    runner.simple_action<interp_component, intrp::Actions::ReceivePoints<
                                               metavars::InterpolationTargetA>>(
        array_index, temporal_id, block_logical_coords, 2_st);
  }

  {
    INFO("Element 6 interpolate");
    // Ensure the volume data got to the interpolator
    const auto& vars_info = get_vars_info(1_st);
    CHECK(vars_info.count(temporal_id) == 1);
    CHECK(vars_info.at(temporal_id).count(element_ids[6]) == 1);

    // Now we should have interpolated data on the 3 elements that have received
    // volume data thus far because our points are now in these elements
    const auto& holder = get_holder(1_st);
    CHECK(holder.temporal_ids_when_data_has_been_interpolated.empty());
    CHECK(holder.infos.size() == 1);
    const auto& info = holder.infos.at(temporal_id);
    CHECK(info.interpolation_is_done_for_these_elements.count(element_ids[6]) ==
          1);
    CHECK(info.block_coord_holders == block_logical_coords);
    CHECK(info.iteration == 2_st);
    CHECK(info.global_offsets.size() == 3);
    CHECK(info.vars.size() == 3);

    // There should be no queued actions and no extra calls to
    // target_receive_vars
    CHECK(runner.is_simple_action_queue_empty<target_component>(0_st));
    CHECK(num_calls_of_target_receive_vars == 1);
  }

  // Send data to second interpolator core from the eighth and final element
  runner.simple_action<
      interp_component,
      intrp::Actions::InterpolatorReceiveVolumeData<Tags::TimeStepId>>(
      1_st, temporal_id, element_ids[7], meshes[7], lapse);

  {
    INFO("Element 7");
    // Ensure the volume data got to the interpolator
    const auto& vars_info = get_vars_info(1_st);
    CHECK(vars_info.count(temporal_id) == 1);
    CHECK(vars_info.at(temporal_id).count(element_ids[6]) == 1);

    // Now we should have interpolated data on all 4 elements and called
    // target_receive_vars. This should have cleaned things up
    const auto& holder = get_holder(1_st);
    CHECK(holder.temporal_ids_when_data_has_been_interpolated.empty());
    CHECK(holder.infos.empty());

    // There should be one queued action; verify this, but not called yet
    CHECK(runner.number_of_queued_simple_actions<target_component>(0_st) == 1);
    CHECK(num_calls_of_target_receive_vars == 1);
    // Invoke the action and check that it's called
    ActionTesting::invoke_queued_simple_action<target_component>(
        make_not_null(&runner), 0_st);
    CHECK(num_calls_of_target_receive_vars == 2);
  }
}
}  // namespace
