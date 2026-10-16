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
#include "Domain/Structure/Topology.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Amr/Criteria/DriveToTarget.hpp"
#include "ParallelAlgorithms/Amr/Criteria/IncreaseResolution.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Tags/Criteria.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Type.hpp"
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
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
class BadCriterion : public amr::Criterion {
 public:
  using options = tmpl::list<>;

  BadCriterion() = default;
  explicit BadCriterion(CkMigrateMessage* msg) : Criterion(msg) {}
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
  static constexpr size_t volume_dim = Metavariables::volume_dim;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<volume_dim>;
  using const_global_cache_tags =
      tmpl::list<amr::Criteria::Tags::Criteria,
                 logging::Tags::Verbosity<amr::OptionTags::AmrGroup>>;
  using simple_tags =
      tmpl::list<domain::Tags::Element<volume_dim>,
                 domain::Tags::Mesh<volume_dim>, amr::Tags::Policies,
                 ::Tags::TimeStepId,
                 evolution::dg::Tags::MortarNextTemporalId<volume_dim>>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 1;
  using component_list = tmpl::list<ElementComponent<Metavariables>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Event, tmpl::list<amr::Events::RefineMesh>>,
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

void test(const Event& event) {
  using element_component = ElementComponent<Metavariables>;

  CHECK(event.needs_evolved_variables());

  const ElementId<1> element_id{0};
  DirectionMap<1, Neighbors<1>> neighbors{};
  neighbors.emplace(
      Direction<1>::upper_xi(),
      Neighbors({ElementId<1>{1}}, OrientationMap<1>::create_aligned()));
  neighbors.emplace(
      Direction<1>::lower_xi(),
      Neighbors({ElementId<1>{2}}, OrientationMap<1>::create_aligned()));
  const Element<1> element{element_id, neighbors,
                           domain::topologies::hypercube<1>};
  const Mesh<1> mesh{std::array{3_st}, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const amr::Policies policies{
      amr::Isotropy::Anisotropic,
      amr::Limits{{{0, 0}}, {{2, 4}}, std::nullopt, std::nullopt, false}, true,
      true};
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

    const amr::Policies error_policies{
        amr::Isotropy::Anisotropic,
        amr::Limits{{{0, 0}}, {{2, 4}}, std::nullopt, std::nullopt, true}, true,
        true};
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

struct B2Metavariables {
  static constexpr size_t volume_dim = 2;
  using component_list = tmpl::list<ElementComponent<B2Metavariables>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Event, tmpl::list<amr::Events::RefineMesh>>,
        tmpl::pair<amr::Criterion, tmpl::list<amr::Criteria::DriveToTarget<
                                       2, amr::Criteria::Type::p>>>>;
  };

  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using projectors =
        tmpl::list<::amr::projectors::DefaultInitialize<
                       Parallel::Tags::GlobalCache<B2Metavariables>>,
                   ::amr::projectors::CopyFromCreatorOrLeaveAsIs<
                       ::domain::Tags::Element<2>>>;
  };
};

void test_b2() {
  INFO("Test enforce topology p-refinement restrictions");
  using element_component = ElementComponent<B2Metavariables>;
  const amr::Events::RefineMesh event{};

  CHECK(event.needs_evolved_variables());

  const ElementId<2> element_id{0};
  const Element<2> element{element_id, DirectionMap<2, Neighbors<2>>{},
                           domain::topologies::disk};
  const Mesh<2> mesh{std::array{3_st, 9_st}, Spectral::bases::disk,
                     Spectral::quadratures::disk};
  const amr::Policies policies{
      amr::Isotropy::Anisotropic,
      amr::Limits{{{0, 0}}, {{2, 4}}, std::nullopt, std::nullopt, false}, true,
      true};
  const Slab slab(3.4, 6.7);
  const TimeStepId time_step_id(true, 5, slab.start());
  DirectionalIdMap<2, TimeStepId> aligned_neighbor_times{};

  std::vector<std::unique_ptr<amr::Criterion>> criteria{};
  // this criteria will choose flags (increase, decrease) to drive the extents
  // towards (5,5), but the topology restriction will change the flags to
  // (increase, increase) which will increase the radial resolution by 1 and
  // the angular resolution by 4
  criteria.emplace_back(
      std::make_unique<amr::Criteria::DriveToTarget<2, amr::Criteria::Type::p>>(
          std::array{5_st, 5_st},
          std::array{amr::Flag::DoNothing, amr::Flag::DoNothing}));
  ActionTesting::MockRuntimeSystem<B2Metavariables> runner{
      {std::move(criteria), ::Verbosity::Debug}};

  ActionTesting::emplace_component_and_initialize<element_component>(
      &runner, element_id,
      {element, mesh, policies, time_step_id, aligned_neighbor_times});
  auto& box = ActionTesting::get_databox<element_component>(
      make_not_null(&runner), element_id);
  auto obs_box = make_observation_box<tmpl::list<>>(make_not_null(&box));
  event.run(make_not_null(&obs_box),
            ActionTesting::cache<element_component>(runner, element_id),
            element_id, std::add_pointer_t<element_component>{},
            {.name = "Time", .value = -1.0});

  const Mesh<2> expected_mesh{std::array{4_st, 13_st}, Spectral::bases::disk,
                              Spectral::quadratures::disk};
  CHECK(
      ActionTesting::get_databox_tag<element_component, domain::Tags::Mesh<2>>(
          runner, element_id) == expected_mesh);
  CHECK(ActionTesting::get_databox_tag<element_component,
                                       domain::Tags::Element<2>>(
            runner, element_id) == element);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Events.RefineMesh", "[Unit][ParallelAlgorithms]") {
  register_factory_classes_with_charm<Metavariables>();
  register_factory_classes_with_charm<B2Metavariables>();
  const amr::Events::RefineMesh event{};
  test(event);
  test(serialize_and_deserialize(event));
  const auto option_event =
      TestHelpers::test_creation<std::unique_ptr<Event>, Metavariables>(
          "RefineMesh\n");
  test(*option_event);
  test(*serialize_and_deserialize(option_event));
  test_b2();
}
