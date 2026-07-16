// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Info.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Amr/Actions/EvaluateRefinementCriteria.hpp"
#include "ParallelAlgorithms/Amr/Criteria/DriveToTarget.hpp"
#include "ParallelAlgorithms/Amr/Criteria/IncreaseResolution.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Tags/Criteria.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Type.hpp"
#include "ParallelAlgorithms/Amr/Policies/Isotropy.hpp"
#include "ParallelAlgorithms/Amr/Policies/Limits.hpp"
#include "ParallelAlgorithms/Amr/Policies/Policies.hpp"
#include "ParallelAlgorithms/Amr/Policies/Tags.hpp"
#include "ParallelAlgorithms/Amr/Protocols/AmrMetavariables.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
auto wants_to_join() {
  return std::make_unique<
      amr::Criteria::DriveToTarget<1, amr::Criteria::Type::h>>(
      std::array{0_st}, std::array{amr::Flag::DoNothing});
}

auto wants_to_split() {
  return std::make_unique<
      amr::Criteria::DriveToTarget<1, amr::Criteria::Type::h>>(
      std::array{8_st}, std::array{amr::Flag::DoNothing});
}

auto wants_to_increase_resolution() {
  return std::make_unique<amr::Criteria::IncreaseResolution<1>>();
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
template <size_t Dim>
class BadCriterion : public amr::Criterion {
 public:
  using options = tmpl::list<>;

  BadCriterion() = default;
  explicit BadCriterion(CkMigrateMessage* msg) : Criterion(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(BadCriterion);  // NOLINT

  amr::Criteria::Type type() override { return amr::Criteria::Type::h; }

  std::string observation_name() override { return "BadCriterion"; }

  using compute_tags_for_observation_box = tmpl::list<>;
  using argument_tags = tmpl::list<>;

  template <typename Metavariables>
  auto operator()(
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Metavariables::volume_dim>& /*element_id*/) const {
    if constexpr (Dim == 1) {
      return std::array{amr::Flag::DecreaseResolution};
    } else if constexpr (Dim == 2) {
      return std::array{amr::Flag::DecreaseResolution,
                        amr::Flag::IncreaseResolution};
    } else {
      return std::array{amr::Flag::DecreaseResolution,
                        amr::Flag::IncreaseResolution, amr::Flag::DoNothing};
    }
  }

  void pup(PUP::er& p) override { Criterion::pup(p); }
};

template <size_t Dim>
PUP::able::PUP_ID BadCriterion<Dim>::my_PUP_ID = 0;  // NOLINT
#pragma GCC diagnostic pop

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  static constexpr size_t volume_dim = Metavariables::volume_dim;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<volume_dim>;
  using const_global_cache_tags =
      tmpl::list<amr::Criteria::Tags::Criteria,
                 amr::Tags::AmrBlocks<volume_dim>, amr::Tags::Policies>;
  using simple_tags =
      tmpl::list<domain::Tags::Element<volume_dim>,
                 domain::Tags::Mesh<volume_dim>, amr::Tags::Info<volume_dim>,
                 amr::Tags::NeighborInfo<volume_dim>>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

template <size_t VolumeDim, bool IgnoreP>
struct Metavariables {
  static constexpr size_t volume_dim = VolumeDim;

  using component_list = tmpl::list<Component<Metavariables>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        amr::Criterion,
        tmpl::list<
            BadCriterion<volume_dim>,
            amr::Criteria::IncreaseResolution<volume_dim>,
            amr::Criteria::DriveToTarget<volume_dim, amr::Criteria::Type::h>,
            amr::Criteria::DriveToTarget<volume_dim, amr::Criteria::Type::p>>>>;
  };

  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using element_array = Component<Metavariables>;
    using projectors = tmpl::list<>;
    static constexpr bool keep_coarse_grids = false;
    static constexpr bool p_refine_only_in_event = IgnoreP;
  };
};

// When AMR is run, the simple action EvaluateAmrCriteria is run on each
// Element.  EvaluateAmrCriteria evaluates the criteria which set its own
// amr::Tags::Info and then calls the simple action UpdateAmrDecision
// on each neighboring Element of the Element sending the Info.
// UpdateAmrDecision checks to see if an Elements Info need to change based on
// the received NeighborInfo (e.g. if an element wants to join, but its
// sibling does not the element must change its decision to do nothing).  If the
// element's Info are changed, then it calls UpdateAmrDecision on its
// neighbors, and the process continues until no Element wants to change its
// decision.   This test manually runs this process on three elements until
// EvaluateAmrCriteria has been called on each Element.  Note in a asynchronus
// parallel environment, it is possible for an Element to execute
// UpdateAmrDecision (triggered by a neighboring Element) prior to executing
// EvaluateAmrCriteria
//
// If IgnoreP is true, only h-refinement criteria are evaluated
template <bool IgnoreP>
void evaluate_criteria(std::vector<std::unique_ptr<amr::Criterion>> criteria,
                       const std::array<amr::Flag, 1> expected_flags) {
  using metavariables = Metavariables<1, IgnoreP>;
  using my_component = Component<metavariables>;
  CAPTURE(expected_flags);
  const bool p_refined =
      (expected_flags == std::array{amr::Flag::IncreaseResolution});
  const ElementId<1> self_id(0, {{{1, 1}}});
  const ElementId<1> lo_id(0, {{{1, 0}}});
  const ElementId<1> up_id(1, {{{1, 0}}});
  const ElementId<1> up_sibling_id(1, {{{1, 1}}});
  const ElementId<1> disabled_block_id(2, {{{0, 0}}});
  const Mesh<1> self_mesh{2_st, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<1> lo_mesh{3_st, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<1> up_mesh{4_st, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<1> up_sibling_mesh{5_st, Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto};
  const Mesh<1> p_self_mesh{3_st, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto};
  const Mesh<1> p_lo_mesh{4_st, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<1> p_up_mesh{5_st, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};

  amr::Info<1> initial_info{std::array{amr::Flag::Undefined}, Mesh<1>{}};
  std::unordered_map<ElementId<1>, amr::Info<1>> initial_neighbor_info;
  const std::vector<size_t> amr_blocks{0, 1};

  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {std::move(criteria), amr_blocks,
       amr::Policies{amr::Isotropy::Anisotropic, amr::Limits{}, true, true}}};

  const Element<1> self(self_id,
                        {{{Direction<1>::lower_xi(),
                           {{lo_id}, OrientationMap<1>::create_aligned()}},
                          {Direction<1>::upper_xi(),
                           {{up_id}, OrientationMap<1>::create_aligned()}}}});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, self_id, {self, self_mesh, initial_info, initial_neighbor_info});

  const Element<1> lo(lo_id,
                      {{{Direction<1>::upper_xi(),
                         {{self_id}, OrientationMap<1>::create_aligned()}}}});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, lo_id, {lo, lo_mesh, initial_info, initial_neighbor_info});

  const Element<1> up(
      up_id, {{{Direction<1>::lower_xi(),
                {{self_id}, OrientationMap<1>::create_aligned()}},
               {Direction<1>::upper_xi(),
                {{up_sibling_id}, OrientationMap<1>::create_aligned()}}}});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, up_id, {up, up_mesh, initial_info, initial_neighbor_info});

  const Element<1> up_sibling(
      up_sibling_id,
      {{{Direction<1>::lower_xi(),
         {{up_id}, OrientationMap<1>::create_aligned()}},
        {Direction<1>::upper_xi(),
         {{disabled_block_id}, OrientationMap<1>::create_aligned()}}}});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, up_sibling_id,
      {up_sibling, up_sibling_mesh, initial_info, initial_neighbor_info});

  const Element<1> disabled_block(
      disabled_block_id,
      {{{Direction<1>::lower_xi(),
         {{up_sibling_id}, OrientationMap<1>::create_aligned()}}}});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, disabled_block_id,
      {disabled_block, up_sibling_mesh, initial_info, initial_neighbor_info});

  runner.set_phase(Parallel::Phase::Testing);

  for (const auto& id : {self_id, lo_id, up_id}) {
    CHECK(ActionTesting::get_databox_tag<my_component, amr::Tags::Info<1>>(
              runner, id) == initial_info);
    CHECK(ActionTesting::get_databox_tag<my_component,
                                         amr::Tags::NeighborInfo<1>>(
              runner, id) == initial_neighbor_info);
    CHECK(
        ActionTesting::is_simple_action_queue_empty<my_component>(runner, id));
  }

  // self runs EvaluateAmrCriteria, queueing UpdateAmrDecision on lo and hi
  ActionTesting::simple_action<my_component,
                               amr::Actions::EvaluateRefinementCriteria>(
      make_not_null(&runner), self_id);

  const amr::Info<1> self_info{expected_flags,
                               p_refined ? p_self_mesh : self_mesh};
  for (const auto& id : {self_id, lo_id, up_id}) {
    CHECK(ActionTesting::get_databox_tag<my_component, amr::Tags::Info<1>>(
              runner, id) == (id == self_id ? self_info : initial_info));
    CHECK(ActionTesting::get_databox_tag<my_component,
                                         amr::Tags::NeighborInfo<1>>(
              runner, id) == initial_neighbor_info);
    CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
              runner, id) == (id == self_id ? 0 : 1));
  }

  // lo runs EvaluateAmrCriteria, queuing UpdateAmrDecision on self
  ActionTesting::simple_action<my_component,
                               amr::Actions::EvaluateRefinementCriteria>(
      make_not_null(&runner), lo_id);

  const amr::Info<1> lo_info{expected_flags, p_refined ? p_lo_mesh : lo_mesh};
  for (const auto& id : {self_id, lo_id, up_id}) {
    CHECK(ActionTesting::get_databox_tag<my_component, amr::Tags::Info<1>>(
              runner, id) ==
          (id == up_id ? initial_info : (id == self_id ? self_info : lo_info)));
    CHECK(ActionTesting::get_databox_tag<my_component,
                                         amr::Tags::NeighborInfo<1>>(
              runner, id) == initial_neighbor_info);
    CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
              runner, id) == 1);
  }

  // up runs UpdateAmrDecision, which queues nothing
  ActionTesting::invoke_queued_simple_action<my_component>(
      make_not_null(&runner), up_id);
  for (const auto& id : {self_id, lo_id, up_id}) {
    CHECK(ActionTesting::get_databox_tag<my_component, amr::Tags::Info<1>>(
              runner, id) ==
          (id == up_id ? initial_info : (id == self_id ? self_info : lo_info)));
    CHECK(ActionTesting::get_databox_tag<my_component,
                                         amr::Tags::NeighborInfo<1>>(runner,
                                                                     id) ==
          (id == up_id
               ? std::unordered_map<ElementId<1>, amr::Info<1>>{{self_id,
                                                                 self_info}}
               : initial_neighbor_info));
    CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
              runner, id) == (id == up_id ? 0 : 1));
  }

  // up runs EvaluateAmrCriteria, queueing UpdateAmrDecision on self
  ActionTesting::simple_action<my_component,
                               amr::Actions::EvaluateRefinementCriteria>(
      make_not_null(&runner), up_id);
  const amr::Info<1> up_info{expected_flags, p_refined ? p_up_mesh : up_mesh};
  for (const auto& id : {self_id, lo_id, up_id}) {
    CHECK(ActionTesting::get_databox_tag<my_component, amr::Tags::Info<1>>(
              runner, id) ==
          (id == up_id ? up_info : (id == self_id ? self_info : lo_info)));
    CHECK(ActionTesting::get_databox_tag<my_component,
                                         amr::Tags::NeighborInfo<1>>(runner,
                                                                     id) ==
          (id == up_id
               ? std::unordered_map<ElementId<1>, amr::Info<1>>{{self_id,
                                                                 self_info}}
               : initial_neighbor_info));
    CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
              runner, id) == (id == self_id ? 2 : (id == lo_id ? 1 : 0)));
  }

  // disabled block runs EvaluateAmrCriteria, which queues nothing
  ActionTesting::simple_action<my_component,
                               amr::Actions::EvaluateRefinementCriteria>(
      make_not_null(&runner), disabled_block_id);
  const amr::Info<1> disabled_block_info{
      std::array<amr::Flag, 1>{amr::Flag::DoNothing}, up_sibling_mesh};
  for (const auto& id : {disabled_block_id}) {
    CHECK(ActionTesting::get_databox_tag<my_component, amr::Tags::Info<1>>(
              runner, id) == disabled_block_info);
    CHECK(ActionTesting::get_databox_tag<my_component,
                                         amr::Tags::NeighborInfo<1>>(
              runner, id) == initial_neighbor_info);
    CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
              runner, id) == 0);
  }
}

void check_split_while_join_is_avoided() {
  using metavariables = Metavariables<2, true>;
  using my_component = Component<metavariables>;

  // The part of action we are testing does not depend upon information
  // from neighbors, so we just use a single Element setup on refinement
  // levels (0, 1)
  const ElementId<2> self_id(0, {{{0, 0}, {1, 1}}});
  const Mesh<2> mesh{2_st, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  amr::Info<2> initial_info{
      std::array{amr::Flag::Undefined, amr::Flag::Undefined}, Mesh<2>{}};
  std::unordered_map<ElementId<2>, amr::Info<2>> initial_neighbor_info{};

  // the refinement criteria wants to drive self to levels (1, 0) so
  // it will return flags (Split, Join).
  std::vector<std::unique_ptr<amr::Criterion>> criteria;
  criteria.emplace_back(
      std::make_unique<amr::Criteria::DriveToTarget<2, amr::Criteria::Type::h>>(
          std::array{1_st, 0_st},
          std::array{amr::Flag::DoNothing, amr::Flag::DoNothing}));

  Parallel::GlobalCache<metavariables> empty_cache{};
  auto databox = db::create<tmpl::list<::domain::Tags::Mesh<2>>>(mesh);
  const ObservationBox<tmpl::list<>,
                       db::DataBox<tmpl::list<::domain::Tags::Mesh<2>>>>
      box{make_not_null(&databox)};
  auto flags_from_criterion =
      criteria.front()->evaluate(box, empty_cache, self_id);
  CHECK(flags_from_criterion == std::array{amr::Flag::Split, amr::Flag::Join});

  // But we do not allow an Element to simultaneously split and join so the
  // action should change the flags to (DoNothing, Split)
  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {std::move(criteria), std::nullopt,
       amr::Policies{amr::Isotropy::Anisotropic, amr::Limits{}, true, true}}};

  const Element<2> self(self_id, {});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, self_id, {self, mesh, initial_info, initial_neighbor_info});

  runner.set_phase(Parallel::Phase::Testing);

  CHECK(ActionTesting::get_databox_tag<my_component, amr::Tags::Info<2>>(
            runner, self_id) == initial_info);
  CHECK(
      ActionTesting::get_databox_tag<my_component, amr::Tags::NeighborInfo<2>>(
          runner, self_id) == initial_neighbor_info);
  CHECK(ActionTesting::is_simple_action_queue_empty<my_component>(runner,
                                                                  self_id));

  // self runs EvaluateAmrCriteria
  ActionTesting::simple_action<my_component,
                               amr::Actions::EvaluateRefinementCriteria>(
      make_not_null(&runner), self_id);

  amr::Info<2> expected_info{std::array{amr::Flag::Split, amr::Flag::DoNothing},
                             mesh};
  CHECK(ActionTesting::get_databox_tag<my_component, amr::Tags::Info<2>>(
            runner, self_id) == expected_info);
  CHECK(
      ActionTesting::get_databox_tag<my_component, amr::Tags::NeighborInfo<2>>(
          runner, self_id) == initial_neighbor_info);
  CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
            runner, self_id) == 0);
}

template <bool IgnoreP>
void check_topology_restrictions_enforced() {
  CAPTURE(IgnoreP);
  using metavariables = Metavariables<3, IgnoreP>;
  using my_component = Component<metavariables>;

  // The part of action we are testing does not depend upon information
  // from neighbors, so we just use a single Element setup on the root
  // refinement level
  const ElementId<3> self_id(0);
  const Mesh<3> mesh{std::array{2_st, 4_st, 7_st},
                     Spectral::bases::spherical_shell<>,
                     Spectral::quadratures::spherical_shell<>};
  const amr::Info<3> initial_info{make_array<3>(amr::Flag::Undefined),
                                  Mesh<3>{}};
  const std::unordered_map<ElementId<3>, amr::Info<3>> initial_neighbor_info{};

  // the refinement criteria wants to drive self to levels (2, 2, 2) so
  // it will return flags (Split, Split, Split).
  std::vector<std::unique_ptr<amr::Criterion>> criteria;
  criteria.emplace_back(
      std::make_unique<amr::Criteria::DriveToTarget<3, amr::Criteria::Type::h>>(
          make_array<3>(2_st), make_array<3>(amr::Flag::DoNothing)));
  // If IgnoreP is false, this criteria will want to drive resolution to
  // (3, 4, 9)
  criteria.emplace_back(
      std::make_unique<amr::Criteria::DriveToTarget<3, amr::Criteria::Type::p>>(
          std::array{3_st, 4_st, 9_st}, make_array<3>(amr::Flag::DoNothing)));

  Parallel::GlobalCache<metavariables> empty_cache{};
  auto databox = db::create<tmpl::list<::domain::Tags::Mesh<3>>>(mesh);
  const ObservationBox<tmpl::list<>,
                       db::DataBox<tmpl::list<::domain::Tags::Mesh<3>>>>
      box{make_not_null(&databox)};
  // Check the raw criteria produce unwanted values
  const auto flags_from_h_criterion =
      criteria.front()->evaluate(box, empty_cache, self_id);
  CHECK(flags_from_h_criterion == make_array<3>(amr::Flag::Split));
  const auto flags_from_p_criterion =
      criteria.back()->evaluate(box, empty_cache, self_id);
  CHECK(flags_from_p_criterion == std::array{amr::Flag::IncreaseResolution,
                                             amr::Flag::DoNothing,
                                             amr::Flag::IncreaseResolution});

  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {std::move(criteria), std::nullopt,
       amr::Policies{amr::Isotropy::Anisotropic, amr::Limits{}, true, true}}};

  const Element<3> self(self_id, {}, domain::topologies::spherical_shell);
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, self_id, {self, mesh, initial_info, initial_neighbor_info});

  runner.set_phase(Parallel::Phase::Testing);

  CHECK(ActionTesting::get_databox_tag<my_component, amr::Tags::Info<3>>(
            runner, self_id) == initial_info);
  CHECK(
      ActionTesting::get_databox_tag<my_component, amr::Tags::NeighborInfo<3>>(
          runner, self_id) == initial_neighbor_info);
  CHECK(ActionTesting::is_simple_action_queue_empty<my_component>(runner,
                                                                  self_id));

  // self runs EvaluateAmrCriteria
  ActionTesting::simple_action<my_component,
                               amr::Actions::EvaluateRefinementCriteria>(
      make_not_null(&runner), self_id);

  // check the action correctly resets flags based on topology
  // If IgonreP is false, IncreaseResolution has higher priority than DoNothing
  const amr::Info<3> expected_info{
      std::array{
          amr::Flag::Split,
          IgnoreP ? amr::Flag::DoNothing : amr::Flag::IncreaseResolution,
          IgnoreP ? amr::Flag::DoNothing : amr::Flag::IncreaseResolution},
      IgnoreP ? mesh
              : Mesh<3>{std::array{2_st, 5_st, 9_st},
                        Spectral::bases::spherical_shell<>,
                        Spectral::quadratures::spherical_shell<>}};
  CHECK(ActionTesting::get_databox_tag<my_component, amr::Tags::Info<3>>(
            runner, self_id) == expected_info);
  CHECK(
      ActionTesting::get_databox_tag<my_component, amr::Tags::NeighborInfo<3>>(
          runner, self_id) == initial_neighbor_info);
  CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
            runner, self_id) == 0);
}
}  //  namespace

SPECTRE_TEST_CASE("Unit.Amr.Actions.EvaluateRefinementCriteria",
                  "[Unit][ParallelAlgorithms]") {
  register_factory_classes_with_charm<Metavariables<3, true>>();
  register_factory_classes_with_charm<Metavariables<3, false>>();
  register_factory_classes_with_charm<Metavariables<2, true>>();
  register_factory_classes_with_charm<Metavariables<1, true>>();
  register_factory_classes_with_charm<Metavariables<1, false>>();
  {
    INFO("No criteria");
    evaluate_criteria<true>(std::vector<std::unique_ptr<amr::Criterion>>{},
                            std::array{amr::Flag::DoNothing});
    evaluate_criteria<false>(std::vector<std::unique_ptr<amr::Criterion>>{},
                             std::array{amr::Flag::DoNothing});
  }
  {
    INFO("Only one p-criteria");
    std::vector<std::unique_ptr<amr::Criterion>> criteria;
    criteria.emplace_back(wants_to_increase_resolution());
    evaluate_criteria<false>(std::move(criteria),
                             std::array{amr::Flag::IncreaseResolution});
  }
  {
    INFO("Only one p-criteria, ignored");
    std::vector<std::unique_ptr<amr::Criterion>> criteria;
    criteria.emplace_back(wants_to_increase_resolution());
    evaluate_criteria<true>(std::move(criteria),
                            std::array{amr::Flag::DoNothing});
  }
  {
    INFO("Should join");
    std::vector<std::unique_ptr<amr::Criterion>> criteria;
    criteria.emplace_back(wants_to_join());
    evaluate_criteria<false>(std::move(criteria), std::array{amr::Flag::Join});
  }
  {
    INFO("Should join, p-ignored");
    std::vector<std::unique_ptr<amr::Criterion>> criteria;
    criteria.emplace_back(wants_to_join());
    criteria.emplace_back(wants_to_increase_resolution());
    evaluate_criteria<true>(std::move(criteria), std::array{amr::Flag::Join});
  }
  {
    INFO("Should split");
    std::vector<std::unique_ptr<amr::Criterion>> criteria;
    criteria.emplace_back(wants_to_split());
    criteria.emplace_back(wants_to_join());
    evaluate_criteria<false>(std::move(criteria), std::array{amr::Flag::Split});
  }
  {
    INFO("Should split, p-ignored");
    std::vector<std::unique_ptr<amr::Criterion>> criteria;
    criteria.emplace_back(wants_to_join());
    criteria.emplace_back(wants_to_increase_resolution());
    criteria.emplace_back(wants_to_split());
    evaluate_criteria<true>(std::move(criteria), std::array{amr::Flag::Split});
  }
#ifdef SPECTRE_DEBUG
  {
    INFO("Check ASSERT triggers");
    std::vector<std::unique_ptr<amr::Criterion>> criteria;
    criteria.emplace_back(std::make_unique<BadCriterion<1>>());

    CHECK_THROWS_WITH(
        (evaluate_criteria<true>(std::move(criteria),
                                 std::array{amr::Flag::DoNothing})),
        Catch::Matchers::ContainsSubstring(
            "requested p-refinement, but claims to be for h-refinement"));
  }
#endif
  check_split_while_join_is_avoided();
  check_topology_restrictions_enforced<true>();
  check_topology_restrictions_enforced<false>();
}
