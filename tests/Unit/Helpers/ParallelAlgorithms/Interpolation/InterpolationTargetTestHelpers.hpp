// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/CleanUpInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolationTargetReceiveVars.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/SendPointsToInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolatedVars.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace intrp {
namespace Actions {
template <typename InterpolationTargetTag>
struct ReceivePoints;
}  // namespace Actions
namespace Tags {
template <typename TemporalId>
struct InterpolatedVarsHolders;
struct NumberOfElements;
}  // namespace Tags
}  // namespace intrp
/// \endcond

namespace InterpTargetTestHelpers {

enum class ValidPoints { All, None, Some };

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
  using const_global_cache_tags = tmpl::flatten<tmpl::append<
      Parallel::get_const_global_cache_tags_from_actions<
          tmpl::list<typename InterpolationTargetTag::compute_target_points>>,
      tmpl::list<domain::Tags::Domain<Metavariables::volume_dim>>>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<intrp::Actions::InitializeInterpolationTarget<
              Metavariables, InterpolationTargetTag>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
};

template <typename InterpolationTargetTag>
struct MockReceivePoints {
  template <
      typename ParallelComponent, typename DbTags, typename Metavariables,
      typename ArrayIndex, size_t VolumeDim,
      Requires<tmpl::list_contains_v<DbTags, ::intrp::Tags::NumberOfElements>> =
          nullptr>
  static void apply(
      db::DataBox<DbTags>& box, Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const typename InterpolationTargetTag::temporal_id::type& temporal_id,
      std::vector<BlockLogicalCoords<VolumeDim>>&& block_coord_holders,
      const size_t iteration = 0_st) {
    db::mutate<intrp::Tags::InterpolatedVarsHolders<Metavariables>>(
        [&temporal_id, &block_coord_holders, &iteration](
            const gsl::not_null<typename intrp::Tags::InterpolatedVarsHolders<
                Metavariables>::type*>
                vars_holders) {
          auto& vars_infos =
              get<intrp::Vars::HolderTag<InterpolationTargetTag,
                                         Metavariables>>(*vars_holders)
                  .infos;

          // Add the target interpolation points at this timestep.
          vars_infos.emplace(std::make_pair(
              temporal_id,
              intrp::Vars::Info<VolumeDim, typename InterpolationTargetTag::
                                               vars_to_interpolate_to_target>{
                  std::move(block_coord_holders), iteration}));
        },
        make_not_null(&box));
  }
};

template <typename Metavariables>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<intrp::Actions::InitializeInterpolator<
              intrp::Tags::VolumeVarsInfo<
                  Metavariables,
                  typename Metavariables::InterpolationTargetA::temporal_id>,
              intrp::Tags::InterpolatedVarsHolders<Metavariables>>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;

  using component_being_mocked = intrp::Interpolator<Metavariables>;
  using replace_these_simple_actions = tmpl::list<intrp::Actions::ReceivePoints<
      typename Metavariables::InterpolationTargetA>>;
  using with_these_simple_actions = tmpl::list<
      MockReceivePoints<typename Metavariables::InterpolationTargetA>>;
};

template <typename InterpolationTargetTag, size_t Dim>
struct MockMetavars {
  static constexpr size_t volume_dim = Dim;
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags = tmpl::list<InterpolationTargetTag>;

  using component_list =
      tmpl::list<InterpTargetTestHelpers::mock_interpolation_target<
          MockMetavars, InterpolationTargetTag>>;
};

template <typename InterpolationTargetTag, size_t Dim,
          typename InterpolationTargetOptionTag, typename BlockCoordHolder>
void test_interpolation_target(
    typename InterpolationTargetOptionTag::type options,
    const BlockCoordHolder& expected_block_coord_holders) {
  using metavars = MockMetavars<InterpolationTargetTag, Dim>;
  using target_component =
      mock_interpolation_target<metavars, InterpolationTargetTag>;
  // Assert that all ComputeTargetPoints conform to the protocol
  static_assert(tt::assert_conforms_to_v<
                typename InterpolationTargetTag::compute_target_points,
                intrp::protocols::ComputeTargetPoints>);

  tuples::TaggedTuple<InterpolationTargetOptionTag,
                      domain::Tags::Domain<metavars::volume_dim>>
      tuple_of_opts{std::move(options), Domain<metavars::volume_dim>{}};

  ActionTesting::MockRuntimeSystem<metavars> runner{std::move(tuple_of_opts)};
  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_component<target_component>(&runner, 0);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<target_component>(make_not_null(&runner), 0);
  }
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  auto& target_box =
      ActionTesting::get_databox<target_component>(make_not_null(&runner), 0);
  const auto& cache = ActionTesting::cache<target_component>(runner, 0_st);

  Slab slab(0.0, 1.0);
  TimeStepId temporal_id(true, 0, ::Time(slab, 0));

  const auto block_coord_holders =
      intrp::InterpolationTarget_detail::block_logical_coords<
          InterpolationTargetTag>(target_box, cache, temporal_id);

  const size_t number_of_points = expected_block_coord_holders.size();
  for (size_t i = 0; i < number_of_points; ++i) {
    if (block_coord_holders[i].has_value()) {
      CHECK(block_coord_holders[i].value().id ==
            expected_block_coord_holders[i].value().id);
      CHECK_ITERABLE_APPROX(block_coord_holders[i].value().data,
                            expected_block_coord_holders[i].value().data);
    }
  }
}
}  // namespace InterpTargetTestHelpers
