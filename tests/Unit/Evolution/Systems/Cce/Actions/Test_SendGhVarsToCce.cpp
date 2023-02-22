// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Evolution/Systems/Cce/Actions/SendGhVarsToCce.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/ParallelAlgorithms/Interpolation/InterpolateOnElementTestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeTargetPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
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
  using simple_tags =
      tmpl::list<::Tags::Time, domain::Tags::Mesh<Metavariables::volume_dim>,
                 ::Tags::Variables<tmpl::list<
                     InterpolateOnElementTestHelpers::Tags::TestSolution>>,
                 intrp::Tags::InterpPointInfo<Metavariables>>;
  using compute_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<Cce::Actions::SendGhVarsToCce<
              typename Metavariables::InterpolationTargetA>>>>;
};

template <typename ElemComponent>
struct initialize_elements_and_queue_simple_actions {
  template <typename InterpPointInfo, typename Runner>
  void operator()(const DomainCreator<3>& domain_creator,
                  const Domain<3>& domain,
                  const std::vector<ElementId<3>>& element_ids,
                  const InterpPointInfo& interp_point_info, Runner& runner,
                  const double time) {
    using elem_component = ElemComponent;

    // Emplace elements.
    for (const auto& element_id : element_ids) {
      // 1. Get vars and mesh
      auto [vars, mesh] =
          InterpolateOnElementTestHelpers::make_volume_data_and_mesh<
              ElemComponent, false>(domain_creator, runner, domain, element_id,
                                    time);

      // 2. emplace element.
      ActionTesting::emplace_component_and_initialize<elem_component>(
          &runner, element_id,
          {time, mesh, std::move(vars), interp_point_info});
    }

    ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

    // Call the action on all the elements.
    for (const auto& element_id : element_ids) {
      ActionTesting::next_action<elem_component>(make_not_null(&runner),
                                                 element_id);
    }
  }
};

struct MockComputeTargetPoints
    : tt::ConformsTo<intrp::protocols::ComputeTargetPoints> {
  using is_sequential = std::false_type;
  using frame = ::Frame::Inertial;
  template <typename Metavariables, typename DbTags, typename TemporalId>
  static tnsr::I<DataVector, 3, Frame::Inertial> points(
      const db::DataBox<DbTags>& /*box*/,
      const tmpl::type_<Metavariables>& /*meta*/,
      const TemporalId& /*temporal_id*/) {
    return tnsr::I<DataVector, 3, Frame::Inertial>{};
  }
};

struct MockPostInterpolationCallback
    : tt::ConformsTo<intrp::protocols::PostInterpolationCallback> {
  template <typename Metavariables, typename DbTags, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const TemporalId& /*temporal_id*/) {}
};

struct InterpolationTargetAImpl
    : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
  using temporal_id = ::Tags::Time;
  using vars_to_interpolate_to_target =
      tmpl::list<InterpolateOnElementTestHelpers::Tags::TestSolution>;
  using compute_target_points = MockComputeTargetPoints;
  using post_interpolation_callback = MockPostInterpolationCallback;
  using compute_items_on_target = tmpl::list<>;
};

struct MockMetavariables {
  using InterpolationTargetA = InterpolationTargetAImpl;
  static constexpr bool use_time_dependent_maps = false;
  static constexpr size_t volume_dim = 3;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<3>>;
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
};

template <typename MockMetavariables>
void run_test() {
  using metavars = MockMetavariables;
  using elem_component = mock_element<metavars>;
  InterpolateOnElementTestHelpers::test_interpolate_on_element<metavars,
                                                               elem_component>(
      initialize_elements_and_queue_simple_actions<elem_component>{});
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Cce.Actions.SendGhVarsToCce",
    "[Unit][Cce]") {
  domain::creators::register_derived_with_charm();
  run_test<MockMetavariables>();
}
}  // namespace
