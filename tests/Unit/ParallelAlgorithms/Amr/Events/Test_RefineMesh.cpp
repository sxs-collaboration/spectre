// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <pup.h>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "IO/Observer/Protocols/ReductionDataFormatter.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Amr/Criteria/DriveToTarget.hpp"
#include "ParallelAlgorithms/Amr/Criteria/IncreaseResolution.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Tags/Criteria.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Type.hpp"
#include "ParallelAlgorithms/Amr/Events/ObserveAmrStats.hpp"
#include "ParallelAlgorithms/Amr/Events/RefineMesh.hpp"
#include "ParallelAlgorithms/Amr/Policies/Isotropy.hpp"
#include "ParallelAlgorithms/Amr/Policies/Limits.hpp"
#include "ParallelAlgorithms/Amr/Policies/Policies.hpp"
#include "ParallelAlgorithms/Amr/Policies/Tags.hpp"
#include "ParallelAlgorithms/Amr/Projectors/CopyFromCreatorOrLeaveAsIs.hpp"
#include "ParallelAlgorithms/Amr/Projectors/DefaultInitialize.hpp"
#include "ParallelAlgorithms/Amr/Protocols/AmrMetavariables.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
static_assert(
    tt::assert_conforms_to_v<amr::Events::detail::FormatAmrStatsOutput,
                             observers::protocols::ReductionDataFormatter>);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
class BadCriterion : public amr::Criterion {
 public:
  using options = tmpl::list<>;

  BadCriterion() = default;
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(BadCriterion);  // NOLINT

  amr::Criteria::Type type() override { return amr::Criteria::Type::p; }

  std::string observation_name() override { return "BadCriterion"; }

  using compute_tags_for_observation_box = tmpl::list<>;
  using argument_tags = tmpl::list<>;

  template <typename Metavariables>
  auto operator()(
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Metavariables::volume_dim>& /*element_id*/) const {
    return std::array{amr::Flag::Split};
  }

  void pup(PUP::er& p) override { Criterion::pup(p); }
};

PUP::able::PUP_ID BadCriterion::my_PUP_ID = 0;  // NOLINT
#pragma GCC diagnostic pop

template <typename Metavariables>
struct ElementComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<1>;
  using const_global_cache_tags =
      tmpl::list<amr::Criteria::Tags::Criteria,
                 logging::Tags::Verbosity<amr::OptionTags::AmrGroup>>;
  using simple_tags =
      tmpl::list<domain::Tags::Element<1>, domain::Tags::Mesh<1>,
                 amr::Tags::Policies, ::Tags::TimeStepId,
                 evolution::dg::Tags::MortarNextTemporalId<1>>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

template <typename Metavariables>
struct MockContributeReductionData {
  template <typename ParallelComponent, typename... DbTags, typename ArrayIndex,
            typename ReductionData, typename Formatter>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationId& /*observation_id*/,
                    Parallel::ArrayComponentId /*sender_array_id*/,
                    const std::string& /*subfile_name*/,
                    const std::vector<std::string>& legend,
                    ReductionData&& reduction_data,
                    std::optional<Formatter>&& /*formatter*/,
                    const bool /*observe_per_core*/) {
    CHECK(legend ==
          std::vector<std::string>{"Time", "NumElements", "TotalNumPoints",
                                   "NumPointsPerDim_0", "MinPointsPerDim_0",
                                   "MaxPointsPerDim_0"});
    // Time
    CHECK(get<0>(reduction_data.data()) == -1.0);
    // Total num elements
    CHECK(get<1>(reduction_data.data()) == 1_st);
    // Total num points
    CHECK(get<2>(reduction_data.data()) == 4_st);
    // Points per dim
    CHECK(get<3>(reduction_data.data()) == std::vector<size_t>{4});
    // Min/max points
    CHECK(get<4>(reduction_data.data()) == std::vector<size_t>{4});
    CHECK(get<5>(reduction_data.data()) == std::vector<size_t>{4});
  }
};

template <typename Metavariables>
struct MockObserverComponent {
  using component_being_mocked = observers::Observer<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<observers::Actions::ContributeReductionData>;
  using with_these_simple_actions =
      tmpl::list<MockContributeReductionData<Metavariables>>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 1;
  using component_list = tmpl::list<ElementComponent<Metavariables>,
                                    MockObserverComponent<Metavariables>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Event, tmpl::list<amr::Events::RefineMesh,
                                     amr::Events::ObserveAmrStats<volume_dim>>>,
        tmpl::pair<
            amr::Criterion,
            tmpl::list<
                BadCriterion, amr::Criteria::IncreaseResolution<1>,
                amr::Criteria::DriveToTarget<1, amr::Criteria::Type::p>,
                amr::Criteria::DriveToTarget<1, amr::Criteria::Type::h>>>>;
  };

  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using projectors =
        tmpl::list<::amr::projectors::DefaultInitialize<
                       Parallel::Tags::GlobalCache<Metavariables>>,
                   ::amr::projectors::CopyFromCreatorOrLeaveAsIs<
                       ::domain::Tags::Element<1>>>;
  };
};

void test(const Event& event, const Event& observe_event) {
  using element_component = ElementComponent<Metavariables>;
  using observer_component = MockObserverComponent<Metavariables>;

  CHECK(event.needs_evolved_variables());

  const ElementId<1> element_id{0};
  DirectionMap<1, Neighbors<1>> neighbors{};
  neighbors.emplace(
      Direction<1>::upper_xi(),
      Neighbors({ElementId<1>{1}}, OrientationMap<1>::create_aligned()));
  neighbors.emplace(
      Direction<1>::lower_xi(),
      Neighbors({ElementId<1>{2}}, OrientationMap<1>::create_aligned()));
  const Element<1> element{element_id, neighbors};
  const Mesh<1> mesh{std::array{3_st}, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const amr::Policies policies{amr::Isotropy::Anisotropic,
                               amr::Limits{0, 0, 3, 5}, true, true};
  const Slab slab(3.4, 6.7);
  const TimeStepId time_step_id(true, 5, slab.start());
  const auto later_time_step_id =
      time_step_id.next_substep(slab.duration() / 2, 0.5);
  DirectionalIdMap<1, TimeStepId> aligned_neighbor_times{};
  for (const auto& [direction, neighbors_in_direction] : neighbors) {
    for (const auto& neighbor : neighbors_in_direction) {
      aligned_neighbor_times.emplace(DirectionalId(direction, neighbor),
                                     time_step_id);
    }
  }

  {
    INFO("Basic function");
    std::vector<std::unique_ptr<amr::Criterion>> criteria{};
    criteria.emplace_back(
        std::make_unique<amr::Criteria::IncreaseResolution<1>>());
    // this should be ignored...
    criteria.emplace_back(
        std::make_unique<
            amr::Criteria::DriveToTarget<1, amr::Criteria::Type::h>>(
            std::array{1_st}, std::array{amr::Flag::DoNothing}));
    ActionTesting::MockRuntimeSystem<Metavariables> runner{
        {std::move(criteria), ::Verbosity::Debug}};
    ActionTesting::emplace_group_component<observer_component>(&runner);

    ActionTesting::emplace_component_and_initialize<element_component>(
        &runner, element_id,
        {element, mesh, policies, time_step_id, aligned_neighbor_times});
    auto& box = ActionTesting::get_databox<element_component>(
        make_not_null(&runner), element_id);
    auto obs_box = make_observation_box<tmpl::list<>>(make_not_null(&box));

    event.run(make_not_null(&obs_box),
              ActionTesting::cache<element_component>(runner, element_id),
              element_id, std::add_pointer_t<element_component>{},
              {"Time", -1.0});
    observe_event.run(
        make_not_null(&obs_box),
        ActionTesting::cache<element_component>(runner, element_id), element_id,
        std::add_pointer_t<element_component>{}, {"Time", -1.0});
    runner.template invoke_queued_simple_action<observer_component>(0);

    const Mesh<1> expected_mesh{std::array{4_st}, Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto};
    CHECK(ActionTesting::get_databox_tag<element_component,
                                         domain::Tags::Mesh<1>>(
              runner, element_id) == expected_mesh);
    CHECK(ActionTesting::get_databox_tag<element_component,
                                         domain::Tags::Element<1>>(
              runner, element_id) == element);
  }

  {
    INFO("Obey policies");
    // Try to drive to smaller number of grid points than we allow
    std::vector<std::unique_ptr<amr::Criterion>> criteria{};
    criteria.emplace_back(
        std::make_unique<
            amr::Criteria::DriveToTarget<1, amr::Criteria::Type::p>>(
            std::array{1_st}, std::array{amr::Flag::DoNothing}));
    ActionTesting::MockRuntimeSystem<Metavariables> runner{
        {std::move(criteria), ::Verbosity::Debug}};
    ActionTesting::emplace_group_component<observer_component>(&runner);

    ActionTesting::emplace_component_and_initialize<element_component>(
        &runner, element_id,
        {element, mesh, policies, time_step_id, aligned_neighbor_times});
    auto& box = ActionTesting::get_databox<element_component>(
        make_not_null(&runner), element_id);
    auto obs_box = make_observation_box<tmpl::list<>>(make_not_null(&box));

    event.run(make_not_null(&obs_box),
              ActionTesting::cache<element_component>(runner, element_id),
              element_id, std::add_pointer_t<element_component>{},
              {"Unused", -1.0});

    const Mesh<1> expected_mesh = mesh;
    CHECK(ActionTesting::get_databox_tag<element_component,
                                         domain::Tags::Mesh<1>>(
              runner, element_id) == expected_mesh);
    CHECK(ActionTesting::get_databox_tag<element_component,
                                         domain::Tags::Element<1>>(
              runner, element_id) == element);

    const amr::Policies error_policies{amr::Isotropy::Anisotropic,
                                       amr::Limits{{{0, 0}}, {{3, 5}}, true},
                                       true, true};
    db::mutate<amr::Tags::Policies>(
        [&](const gsl::not_null<amr::Policies*> box_policies) {
          *box_policies = error_policies;
        },
        make_not_null(&box));

    CHECK_THROWS_WITH(
        (event.run(make_not_null(&obs_box),
                   ActionTesting::cache<element_component>(runner, element_id),
                   element_id, std::add_pointer_t<element_component>{},
                   {"Unused", -1.0})),
        Catch::Matchers::ContainsSubstring(
            "Tried refining beyond the AMR limits in element"));
  }

  {
    INFO("Test unaligned error");
    std::vector<std::unique_ptr<amr::Criterion>> criteria{};
    criteria.emplace_back(
        std::make_unique<amr::Criteria::IncreaseResolution<1>>());
    ActionTesting::MockRuntimeSystem<Metavariables> runner{
        {std::move(criteria), ::Verbosity::Debug}};

    auto neighbor_times = aligned_neighbor_times;
    neighbor_times.begin()->second = later_time_step_id;
    ActionTesting::emplace_component_and_initialize<element_component>(
        &runner, element_id,
        {element, mesh, policies, time_step_id, neighbor_times});
    auto& box = ActionTesting::get_databox<element_component>(
        make_not_null(&runner), element_id);
    auto obs_box = make_observation_box<tmpl::list<>>(make_not_null(&box));

    CHECK_THROWS_WITH(
        event.run(make_not_null(&obs_box),
                  ActionTesting::cache<element_component>(runner, element_id),
                  element_id, std::add_pointer_t<element_component>{},
                  {"Unused", -1.0}),
        Catch::Matchers::ContainsSubstring(
            "Cannot refine mesh when not temporally aligned with neighbors."));
  }

#ifdef SPECTRE_DEBUG
  {
    INFO("Test h-refinement error");
    // Try to use a bad criterion
    std::vector<std::unique_ptr<amr::Criterion>> criteria{};
    criteria.emplace_back(std::make_unique<BadCriterion>());
    ActionTesting::MockRuntimeSystem<Metavariables> runner{
        {std::move(criteria), ::Verbosity::Debug}};
    ActionTesting::emplace_group_component<observer_component>(&runner);

    ActionTesting::emplace_component_and_initialize<element_component>(
        &runner, element_id,
        {element, mesh, policies, time_step_id, aligned_neighbor_times});
    auto& box = ActionTesting::get_databox<element_component>(
        make_not_null(&runner), element_id);
    auto obs_box = make_observation_box<tmpl::list<>>(make_not_null(&box));

    CHECK_THROWS_WITH(
        (event.run(make_not_null(&obs_box),
                   ActionTesting::cache<element_component>(runner, element_id),
                   element_id, std::add_pointer_t<element_component>{},
                   {"Unused", -1.0})),
        Catch::Matchers::ContainsSubstring(
            "requested h-refinement, but claims to be for p-refinement"));
  }
#endif
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Events.RefineMesh", "[Unit][ParallelAlgorithms]") {
  register_factory_classes_with_charm<Metavariables>();
  const amr::Events::RefineMesh event{};
  const amr::Events::ObserveAmrStats<1> observe_event{true, false};
  test(event, observe_event);
  test(serialize_and_deserialize(event),
       serialize_and_deserialize(observe_event));
  const auto option_event =
      TestHelpers::test_creation<std::unique_ptr<Event>, Metavariables>(
          "RefineMesh\n");
  const auto option_observe_event =
      TestHelpers::test_creation<std::unique_ptr<Event>, Metavariables>(
          "ObserveAmrStats:\n"
          "  PrintToTerminal: True\n"
          "  ObservePerCore: False");
  test(*option_event, *option_observe_event);
  test(*serialize_and_deserialize(option_event),
       *serialize_and_deserialize(option_observe_event));
}
