// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/Interpolation/InterpolateOnElementTestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/Events/InterpolateWithoutInterpComponent.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <typename Metavariables>
struct mock_element {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Metavariables::volume_dim>;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
  using initial_databox = db::compute_databox_type<db::AddSimpleTags<>>;
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
      ActionTesting::emplace_component<elem_component>(&runner, element_id);
      ActionTesting::next_action<elem_component>(make_not_null(&runner),
                                                 element_id);
    }
    ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);

    // Create event.
    intrp::Events::InterpolateWithoutInterpComponent<
        metavars::volume_dim, typename metavars::InterpolationTargetA, metavars,
        tmpl::list<InterpolateOnElementTestHelpers::Tags::TestSolution>>
        event{};

    // Run event on all elements.
    for (const auto& element_id : element_ids) {
      // 1. Get vars and mesh
      const auto& [vars, mesh] =
          InterpolateOnElementTestHelpers::make_volume_data_and_mesh(
              domain_creator, domain, element_id);

      // 2. Make a box
      const auto box = db::create<
          db::AddSimpleTags<typename metavars::temporal_id,
                            intrp::Tags::InterpPointInfo<metavars>,
                            domain::Tags::Mesh<metavars::volume_dim>,
                            ::Tags::Variables<typename std::remove_reference_t<
                                decltype(vars)>::tags_list>>>(
          temporal_id, interp_point_info, mesh, vars);

      // 3. Run the event.  This will invoke simple actions on
      // InterpolationTarget.
      event.run(box, runner.cache(), element_id,
                std::add_pointer_t<elem_component>{});
    }
  }
};

template <bool HaveComputeItemsOnSource>
struct MockMetavariables {
  struct InterpolationTargetA {
    using vars_to_interpolate_to_target = tmpl::list<tmpl::conditional_t<
        HaveComputeItemsOnSource,
        InterpolateOnElementTestHelpers::Tags::MultiplyByTwo,
        InterpolateOnElementTestHelpers::Tags::TestSolution>>;
    using compute_items_on_source = tmpl::conditional_t<
        HaveComputeItemsOnSource,
        tmpl::list<
            InterpolateOnElementTestHelpers::Tags::MultiplyByTwoComputeItem>,
        tmpl::list<>>;
  };
  using temporal_id = ::Tags::TimeStepId;
  static constexpr size_t volume_dim = 3;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;

  using component_list =
      tmpl::list<InterpolateOnElementTestHelpers::mock_interpolation_target<
                     MockMetavariables, InterpolationTargetA>,
                 mock_element<MockMetavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

template <typename MockMetavariables>
void run_test() noexcept {
  using metavars = MockMetavariables;
  using elem_component = mock_element<metavars>;
  InterpolateOnElementTestHelpers::test_interpolate_on_element<metavars,
                                                               elem_component>(
      initialize_elements_and_queue_simple_actions<metavars, elem_component>{});
}

SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Interpolator.InterpolateEventNoInterpolator",
    "[Unit]") {
  run_test<MockMetavariables<false>>();
  run_test<MockMetavariables<true>>();
}
}  // namespace
