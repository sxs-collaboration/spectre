// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/ParallelAlgorithms/Interpolation/InterpolateOnElementTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/Interpolation/Events/InterpolateWithoutInterpComponent.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
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
                  const TemporalId& temporal_id) {
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
    typename metavars::event event{};

    CHECK(event.needs_evolved_variables());

    // Run event on all elements.
    for (const auto& element_id : element_ids) {
      // 1. Get vars and mesh
      const auto& [vars, mesh] =
          InterpolateOnElementTestHelpers::make_volume_data_and_mesh(
              domain_creator, domain, element_id);

      // 2. Make a box
      const auto box = db::create<db::AddSimpleTags<
          Parallel::Tags::MetavariablesImpl<metavars>,
          typename metavars::InterpolationTargetA::temporal_id,
          intrp::Tags::InterpPointInfo<metavars>,
          domain::Tags::Mesh<metavars::volume_dim>,
          ::Tags::Variables<
              typename std::remove_reference_t<decltype(vars)>::tags_list>>>(
          metavars{}, temporal_id, interp_point_info, mesh, vars);

      // 3. Run the event.  This will invoke simple actions on
      // InterpolationTarget.
      event.run(box,
                ActionTesting::cache<elem_component>(runner, element_id),
                element_id, std::add_pointer_t<elem_component>{});
    }
  }
};

template <bool HaveComputeItemsOnSource>
struct MockMetavariables {
  struct InterpolationTargetA {
    using temporal_id = ::Tags::Time;
    using vars_to_interpolate_to_target = tmpl::list<tmpl::conditional_t<
        HaveComputeItemsOnSource,
        InterpolateOnElementTestHelpers::Tags::MultiplyByTwo,
        InterpolateOnElementTestHelpers::Tags::TestSolution>>;
    using compute_items_on_source = tmpl::conditional_t<
        HaveComputeItemsOnSource,
        tmpl::list<InterpolateOnElementTestHelpers::Tags::MultiplyByTwoCompute>,
        tmpl::list<>>;
  };
  static constexpr size_t volume_dim = 3;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;

  using component_list =
      tmpl::list<InterpolateOnElementTestHelpers::mock_interpolation_target<
                     MockMetavariables, InterpolationTargetA>,
                 mock_element<MockMetavariables>>;

  using event = intrp::Events::InterpolateWithoutInterpComponent<
      volume_dim, InterpolationTargetA, MockMetavariables,
      tmpl::list<InterpolateOnElementTestHelpers::Tags::TestSolution>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<Event, tmpl::list<event>>>;
  };

  enum class Phase { Initialization, Testing, Exit };
};

template <typename MockMetavariables>
void run_test() {
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
