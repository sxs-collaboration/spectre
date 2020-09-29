// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/Interpolation/InterpolateOnElementTestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/Actions/InterpolateToTarget.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <typename Metavariables, bool AddComputeItemToBox>
struct mock_element {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Metavariables::volume_dim>;
  using simple_tags =
      tmpl::list<typename Metavariables::temporal_id,
                 domain::Tags::Mesh<Metavariables::volume_dim>,
                 ::Tags::Variables<tmpl::list<
                     InterpolateOnElementTestHelpers::Tags::TestSolution>>,
                 intrp::Tags::InterpPointInfo<Metavariables>>;
  using compute_tags = tmpl::conditional_t<
      AddComputeItemToBox,
      tmpl::list<InterpolateOnElementTestHelpers::Tags::MultiplyByTwoCompute>,
      tmpl::list<>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<intrp::Actions::InterpolateToTarget<
              typename Metavariables::InterpolationTargetA>>>>;
};

template <typename Metavariables, typename ElemComponent>
struct initialize_elements_and_queue_simple_actions {
  template <typename InterpPointInfo, typename Runner, typename TemporalId>
  void operator()(const DomainCreator<3>& domain_creator,
                  const Domain<3>& domain,
                  const std::vector<ElementId<3>>& element_ids,
                  const InterpPointInfo& interp_point_info, Runner& runner,
                  const TemporalId& temporal_id) noexcept {
    using metavars = Metavariables;
    using elem_component = ElemComponent;

    // Emplace elements.
    for (const auto& element_id : element_ids) {
      // 1. Get vars and mesh
      auto [vars, mesh] =
          InterpolateOnElementTestHelpers::make_volume_data_and_mesh(
              domain_creator, domain, element_id);

      // 2. emplace element.
      ActionTesting::emplace_component_and_initialize<elem_component>(
          &runner, element_id,
          {temporal_id, mesh, std::move(vars), interp_point_info});
    }

    ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);

    // Call the action on all the elements.
    for (const auto& element_id : element_ids) {
      ActionTesting::next_action<elem_component>(make_not_null(&runner),
                                                 element_id);
      if (Metavariables::has_should_interpolate and
          not Metavariables::should_interpolate_value) {
        CHECK(ActionTesting::is_simple_action_queue_empty<
              InterpolateOnElementTestHelpers::mock_interpolation_target<
                  Metavariables, typename Metavariables::InterpolationTargetA>>(
            runner, 0));
      }
    }
  }
};

template <bool HaveComputeItemsOnSource, bool ShouldInterpolate,
          bool HasShouldInterpolateLocal>
struct InterpolationTargetAImpl;

template <bool HaveComputeItemsOnSource, bool ShouldInterpolate>
struct InterpolationTargetAImpl<HaveComputeItemsOnSource, ShouldInterpolate,
                                true> {
  using vars_to_interpolate_to_target = tmpl::list<
      tmpl::conditional_t<HaveComputeItemsOnSource,
                          InterpolateOnElementTestHelpers::Tags::MultiplyByTwo,
                          InterpolateOnElementTestHelpers::Tags::TestSolution>>;
  using compute_items_on_source = tmpl::conditional_t<
      HaveComputeItemsOnSource,
      tmpl::list<InterpolateOnElementTestHelpers::Tags::MultiplyByTwoCompute>,
      tmpl::list<>>;
  template <typename DbTagList>
  static bool should_interpolate(
      const db::DataBox<DbTagList>& /*box*/) noexcept {
    return ShouldInterpolate;
  }
};

template <bool HaveComputeItemsOnSource, bool ShouldInterpolate>
struct InterpolationTargetAImpl<HaveComputeItemsOnSource, ShouldInterpolate,
                                false> {
  using vars_to_interpolate_to_target = tmpl::list<
      tmpl::conditional_t<HaveComputeItemsOnSource,
                          InterpolateOnElementTestHelpers::Tags::MultiplyByTwo,
                          InterpolateOnElementTestHelpers::Tags::TestSolution>>;
  using compute_items_on_source = tmpl::conditional_t<
      HaveComputeItemsOnSource,
      tmpl::list<InterpolateOnElementTestHelpers::Tags::MultiplyByTwoCompute>,
      tmpl::list<>>;
};

template <bool HaveComputeItemsOnSource, bool AddComputeItemToBox,
          bool HasShouldInterpolateFunction, bool ShouldInterpolate>
struct MockMetavariables {
  static constexpr bool add_compute_item_to_box = AddComputeItemToBox;
  static constexpr bool should_interpolate_value = ShouldInterpolate;
  static constexpr bool has_should_interpolate = HasShouldInterpolateFunction;

  using InterpolationTargetA =
      InterpolationTargetAImpl<HaveComputeItemsOnSource, ShouldInterpolate,
                               HasShouldInterpolateFunction>;

  using temporal_id = ::Tags::TimeStepId;
  static constexpr size_t volume_dim = 3;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;

  using component_list =
      tmpl::list<InterpolateOnElementTestHelpers::mock_interpolation_target<
                     MockMetavariables, InterpolationTargetA>,
                 mock_element<MockMetavariables, AddComputeItemToBox>>;
  enum class Phase { Initialization, Testing, Exit };
};

template <typename MockMetavariables>
void run_test() noexcept {
  using metavars = MockMetavariables;
  using elem_component =
      mock_element<metavars, metavars::add_compute_item_to_box>;
  InterpolateOnElementTestHelpers::test_interpolate_on_element<metavars,
                                                               elem_component>(
      initialize_elements_and_queue_simple_actions<metavars, elem_component>{});
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.InterpolateToTarget",
                  "[Unit]") {
  // should interpolate:
  run_test<MockMetavariables<false, false, true, true>>();
  run_test<MockMetavariables<true, true, true, true>>();
  run_test<MockMetavariables<true, false, true, true>>();
  run_test<MockMetavariables<false, true, true, true>>();
  // should not interpolate:
  run_test<MockMetavariables<false, false, true, false>>();
  run_test<MockMetavariables<true, true, true, false>>();
  run_test<MockMetavariables<true, false, true, false>>();
  run_test<MockMetavariables<false, true, true, false>>();
  // no should_interpolate function:
  run_test<MockMetavariables<false, false, false, false>>();
  run_test<MockMetavariables<true, true, false, false>>();
  run_test<MockMetavariables<true, false, false, false>>();
  run_test<MockMetavariables<false, true, false, false>>();
}
}  // namespace
