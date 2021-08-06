// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Evolution/Systems/Cce/Actions/InterpolateDuringSelfStart.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/Interpolation/InterpolateOnElementTestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <typename Metavariables>
struct mock_element {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Metavariables::volume_dim>;
  using simple_tags =
      tmpl::list<::Tags::Time, domain::Tags::Mesh<Metavariables::volume_dim>,
                 ::Tags::Variables<tmpl::list<
                     InterpolateOnElementTestHelpers::Tags::TestSolution>>,
                 intrp::Tags::InterpPointInfo<Metavariables>>;
  using compute_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<Cce::Actions::InterpolateDuringSelfStart<
              typename Metavariables::InterpolationTargetA>>>>;
};

template <typename Metavariables, typename ElemComponent>
struct initialize_elements_and_queue_simple_actions {
  template <typename InterpPointInfo, typename Runner>
  void operator()(const DomainCreator<3>& domain_creator,
                  const Domain<3>& domain,
                  const std::vector<ElementId<3>>& element_ids,
                  const InterpPointInfo& interp_point_info, Runner& runner,
                  const double time) noexcept {
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
          {time, mesh, std::move(vars), interp_point_info});
    }

    ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);

    // Call the action on all the elements.
    for (const auto& element_id : element_ids) {
      ActionTesting::next_action<elem_component>(make_not_null(&runner),
                                                 element_id);
    }
  }
};

struct InterpolationTargetAImpl {
  using temporal_id = ::Tags::Time;
  using vars_to_interpolate_to_target =
      tmpl::list<InterpolateOnElementTestHelpers::Tags::TestSolution>;
  using compute_items_on_source = tmpl::list<>;
};

struct MockMetavariables {
  using InterpolationTargetA = InterpolationTargetAImpl;

  static constexpr size_t volume_dim = 3;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        Event, tmpl::flatten<
                   tmpl::list<intrp::Events::InterpolateWithoutInterpComponent<
                       volume_dim, InterpolationTargetA, MockMetavariables,
                       typename InterpolationTargetAImpl::
                           vars_to_interpolate_to_target>>>>>;
  };

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
    "Unit.Evolution.Systems.Cce.Actions.InterpolateDuringSelfStart",
    "[Unit][Cce]") {
  run_test<MockMetavariables>();
}
}  // namespace
