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
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Framework/ActionTesting.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

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
          typename InterpolationTargetOptionTag, typename BlockCoordHolder,
          typename... ExtraCacheObjects>
void test_interpolation_target(
    typename InterpolationTargetOptionTag::type options,
    const BlockCoordHolder& expected_block_coord_holders,
    const ExtraCacheObjects&... extra_cache_objects) {
  using metavars = MockMetavars<InterpolationTargetTag, Dim>;
  using target_component =
      mock_interpolation_target<metavars, InterpolationTargetTag>;
  // Assert that all ComputeTargetPoints conform to the protocol
  static_assert(tt::assert_conforms_to_v<
                typename InterpolationTargetTag::compute_target_points,
                intrp::protocols::ComputeTargetPoints>);

  ActionTesting::MockRuntimeSystem<metavars> runner{
      {extra_cache_objects..., std::move(options),
       Domain<metavars::volume_dim>{}, ::Verbosity::Silent}};
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
