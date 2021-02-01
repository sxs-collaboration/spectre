// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/ApplyOperator.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Actions/Goto.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Actions/SetData.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "Utilities/Literals.hpp"

// #include "Parallel/Printf.hpp"

namespace {

template <typename Tag>
struct DgOperatorAppliedTo : db::SimpleTag, db::PrefixTag {
  using type = typename Tag::type;
  using tag = Tag;
};

template <typename Tag>
struct Var : db::SimpleTag, db::PrefixTag {
  using type = typename Tag::type;
  using tag = Tag;
};

struct TemporalIdTag : db::SimpleTag {
  using type = size_t;
};

struct IncrementTemporalId {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<TemporalIdTag>(
        make_not_null(&box),
        [](const auto temporal_id) noexcept { (*temporal_id)++; });
    return {std::move(box)};
  }
};

// Label to indicate the start of the apply-operator actions
struct ApplyOperatorStart {};

template <typename System, bool Linearized, typename Metavariables>
struct ElementArray {
  static constexpr size_t Dim = System::volume_dim;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<Dim>>;

  using primal_vars = db::wrap_tags_in<Var, typename System::primal_fields>;
  using auxiliary_vars =
      db::wrap_tags_in<Var, typename System::auxiliary_fields>;
  using vars_tag = ::Tags::Variables<primal_vars>;
  using auxiliary_vars_tag = ::Tags::Variables<auxiliary_vars>;
  using operator_applied_to_vars_tag =
      ::Tags::Variables<db::wrap_tags_in<DgOperatorAppliedTo, primal_vars>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<
                  tmpl::list<domain::Tags::InitialRefinementLevels<Dim>,
                             domain::Tags::InitialExtents<Dim>, TemporalIdTag>>,
              ::Actions::SetupDataBox, ::dg::Actions::InitializeDomain<Dim>,
              ::elliptic::dg::Actions::initialize_operator<
                  System, TemporalIdTag, vars_tag, operator_applied_to_vars_tag,
                  auxiliary_vars_tag>,
              Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<::Actions::Label<ApplyOperatorStart>,
                     ::elliptic::dg::Actions::apply_operator<
                         System, Linearized, TemporalIdTag, vars_tag,
                         operator_applied_to_vars_tag, auxiliary_vars_tag>,
                     IncrementTemporalId, Parallel::Actions::TerminatePhase>>>;
};

template <typename System, bool Linearized>
struct Metavariables {
  using element_array = ElementArray<System, Linearized, Metavariables>;
  using component_list = tmpl::list<element_array>;
  enum class Phase { Initialization, Testing, Exit };
};

template <typename System, bool Linearized, size_t Dim = System::volume_dim,
          typename Metavars = Metavariables<System, Linearized>,
          typename ElementArray = typename Metavars::element_array>
void test_dg_operator(
    const DomainCreator<Dim>& domain_creator,
    const std::vector<std::tuple<
        std::unordered_map<ElementId<Dim>,
                           typename ElementArray::vars_tag::type>,
        std::unordered_map<ElementId<Dim>,
                           typename ElementArray::auxiliary_vars_tag::type>,
        std::unordered_map<
            ElementId<Dim>,
            typename ElementArray::operator_applied_to_vars_tag::type>>>&
        tests_data) {
  using element_array = ElementArray;
  using vars_tag = typename element_array::vars_tag;
  using auxiliary_vars_tag = typename element_array::auxiliary_vars_tag;
  using operator_applied_to_vars_tag =
      typename element_array::operator_applied_to_vars_tag;
  using Vars = typename vars_tag::type;
  using AuxiliaryVars = typename auxiliary_vars_tag::type;
  using OperatorAppliedToVars = typename operator_applied_to_vars_tag::type;

  // Get a list of all elements in the domain
  auto domain = domain_creator.create_domain();
  const auto initial_ref_levs = domain_creator.initial_refinement_levels();
  const auto initial_extents = domain_creator.initial_extents();
  std::unordered_set<ElementId<Dim>> all_element_ids{};
  for (const auto& block : domain.blocks()) {
    auto block_element_ids =
        initial_element_ids(block.id(), initial_ref_levs[block.id()]);
    for (auto& element_id : block_element_ids) {
      all_element_ids.insert(std::move(element_id));
    }
  }

  // Initialize all elements in the domain
  ActionTesting::MockRuntimeSystem<Metavars> runner{
      tuples::TaggedTuple<domain::Tags::Domain<Dim>,
                          ::elliptic::dg::Tags::PenaltyParameter>{
          std::move(domain), 1.5}};
  for (const auto& element_id : all_element_ids) {
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id, {initial_ref_levs, initial_extents, 0_st});
    while (
        not ActionTesting::get_terminate<element_array>(runner, element_id)) {
      ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                element_id);
    }
  }
  ActionTesting::set_phase(make_not_null(&runner), Metavars::Phase::Testing);

  // DataBox shortcuts
  const auto get_tag = [&runner](
                           auto tag_v,
                           const ElementId<Dim>& element_id) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                              element_id);
  };
  const auto set_tag = [&runner](auto tag_v, const auto& value,
                                 const ElementId<Dim>& element_id) {
    using tag = std::decay_t<decltype(tag_v)>;
    ActionTesting::simple_action<element_array,
                                 ::Actions::SetData<tmpl::list<tag>>>(
        make_not_null(&runner), element_id, value);
  };

  const auto apply_operator_and_check_result =
      [&runner, &all_element_ids, &get_tag, &set_tag](
          const std::unordered_map<ElementId<Dim>, Vars>& all_vars,
          const std::unordered_map<ElementId<Dim>, AuxiliaryVars>&
              all_expected_aux_vars,
          const std::unordered_map<ElementId<Dim>, OperatorAppliedToVars>&
              all_expected_operator_applied_to_vars) {
        // Set variables data on central element and on its neighbors
        for (const auto& element_id : all_element_ids) {
          const auto vars = all_vars.find(element_id);
          if (vars == all_vars.end()) {
            const size_t num_points =
                get_tag(domain::Tags::Mesh<Dim>{}, element_id)
                    .number_of_grid_points();
            set_tag(vars_tag{}, Vars{num_points, 0.}, element_id);
          } else {
            set_tag(vars_tag{}, vars->second, element_id);
          }
        }

        // Apply DG operator
        // 1. Prepare elements and send data
        for (const auto& element_id : all_element_ids) {
          runner.template mock_distributed_objects<element_array>()
              .at(element_id)
              .set_terminate(false);
          // Parallel::printf("Prepare %s:\n", element_id);
          while (ActionTesting::is_ready<element_array>(runner, element_id) and
                 not ActionTesting::get_terminate<element_array>(runner,
                                                                 element_id)) {
            ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                      element_id);
          }
        }
        // 2. Receive data and apply operator
        for (const auto& element_id : all_element_ids) {
          // Parallel::printf("Apply %s:\n", element_id);
          while (not ActionTesting::get_terminate<element_array>(runner,
                                                                 element_id)) {
            ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                      element_id);
          }
        }

        // Check result
        {
          INFO("Auxiliary variables");
          for (const auto& [element_id, expected_aux_vars] :
               all_expected_aux_vars) {
            CAPTURE(element_id);
            const auto& auxiliary_vars =
                get_tag(auxiliary_vars_tag{}, element_id);
            CHECK_VARIABLES_APPROX(auxiliary_vars, expected_aux_vars);
          }
        }
        {
          INFO("Operator applied to variables");
          for (const auto& [element_id, expected_operator_applied_to_vars] :
               all_expected_operator_applied_to_vars) {
            CAPTURE(element_id);
            const auto& operator_applied_to_vars =
                get_tag(operator_applied_to_vars_tag{}, element_id);
            CHECK_VARIABLES_APPROX(operator_applied_to_vars,
                                   expected_operator_applied_to_vars);
          }
        }
      };

  // Test that A(0) = 0
  std::unordered_map<ElementId<Dim>, AuxiliaryVars> all_zero_aux_vars{};
  std::unordered_map<ElementId<Dim>, OperatorAppliedToVars>
      all_zero_operator_vars{};
  for (const auto& element_id : all_element_ids) {
    const size_t num_points =
        get_tag(domain::Tags::Mesh<Dim>{}, element_id).number_of_grid_points();
    all_zero_aux_vars[element_id] = AuxiliaryVars{num_points, 0.};
    all_zero_operator_vars[element_id] = OperatorAppliedToVars{num_points, 0.};
  }
  apply_operator_and_check_result({}, all_zero_aux_vars,
                                  all_zero_operator_vars);

  // Run tests with specified data in sequence
  for (const auto& test_data : tests_data) {
    std::apply(apply_operator_and_check_result, test_data);
  }
  // CHECK(get_tag(TemporalIdTag{}, element_id) == tests_data.size());
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.DG.Operator", "[Unit][Elliptic]") {
  domain::creators::register_derived_with_charm();
  // This is what the tests below check:
  //
  // - The DG operator passes some basic consistency checks, e.g. A(0) = 0.
  // - It produces an expected output from a random (but hard-coded) set of
  //   variables. This is an important regression test, i.e. it ensures the
  //   output does not change with optimizations etc.
  // - It runs on various domain geometries.
  // - The free functions, actions and initialization actions are compatible.
  {
    INFO("1D");
    // Domain decomposition:
    // [ | | | ]-> xi
    const domain::creators::Interval domain_creator{
        {{-0.5}}, {{1.5}}, {{2}}, {{4}}, {{false}}, nullptr};
    const ElementId<1> left_id{0, {{{2, 0}}}};
    const ElementId<1> right_id{0, {{{2, 1}}}};
    using system =
        Poisson::FirstOrderSystem<1, Poisson::Geometry::FlatCartesian>;
    using Vars = Variables<tmpl::list<Var<Poisson::Tags::Field>>>;
    using AuxVars = Variables<tmpl::list<Var<::Tags::deriv<
        Poisson::Tags::Field, tmpl::size_t<1>, Frame::Inertial>>>>;
    using OperatorVars =
        Variables<tmpl::list<DgOperatorAppliedTo<Var<Poisson::Tags::Field>>>>;
    Vars vars_rnd_left{4};
    get(get<Var<Poisson::Tags::Field>>(vars_rnd_left)) =
        DataVector{0.6964691855978616, 0.28613933495037946, 0.2268514535642031,
                   0.5513147690828912};
    Vars vars_rnd_right{4};
    get(get<Var<Poisson::Tags::Field>>(vars_rnd_right)) =
        DataVector{0.7194689697855631, 0.42310646012446096, 0.9807641983846155,
                   0.6848297385848633};
    AuxVars expected_aux_vars_rnd{4};
    get<0>(get<Var<::Tags::deriv<Poisson::Tags::Field, tmpl::size_t<1>,
                                 Frame::Inertial>>>(expected_aux_vars_rnd)) =
        DataVector{12.688072373020212, -1.9207736184862605, 1.3653213194134493,
                   5.338593988765293};
    OperatorVars expected_operator_vars_rnd{4};
    get(get<DgOperatorAppliedTo<Var<Poisson::Tags::Field>>>(
        expected_operator_vars_rnd)) =
        DataVector{1785.7616039198579, 41.55242721423977, -41.54933376905333,
                   -80.83628748342855};
    test_dg_operator<system, true>(
        domain_creator, {{{{left_id, std::move(vars_rnd_left)},
                           {right_id, std::move(vars_rnd_right)}},
                          {{left_id, std::move(expected_aux_vars_rnd)}},
                          {{left_id, std::move(expected_operator_vars_rnd)}}}});
  }
  {
    INFO("2D");
    // Domain decomposition:
    // ^ eta
    // +-+-+> xi
    // | | |
    // +-+-+
    // | | |
    // +-+-+
    const domain::creators::Rectangle domain_creator{
        {{-0.5, 0.}}, {{1.5, 1.}}, {{false, false}}, {{1, 1}}, {{3, 2}}};
    const ElementId<2> northwest_id{0, {{{1, 0}, {1, 1}}}};
    const ElementId<2> southwest_id{0, {{{1, 0}, {1, 0}}}};
    const ElementId<2> northeast_id{0, {{{1, 1}, {1, 1}}}};
    using system =
        Poisson::FirstOrderSystem<2, Poisson::Geometry::FlatCartesian>;
    using Vars = Variables<tmpl::list<Var<Poisson::Tags::Field>>>;
    using AuxVars = Variables<tmpl::list<Var<::Tags::deriv<
        Poisson::Tags::Field, tmpl::size_t<2>, Frame::Inertial>>>>;
    using OperatorVars =
        Variables<tmpl::list<DgOperatorAppliedTo<Var<Poisson::Tags::Field>>>>;
    Vars vars_rnd_northwest{6};
    get(get<Var<Poisson::Tags::Field>>(vars_rnd_northwest)) =
        DataVector{0.9807641983846155, 0.6848297385848633, 0.48093190148436094,
                   0.3921175181941505, 0.3431780161508694, 0.7290497073840416};
    Vars vars_rnd_southwest{6};
    get(get<Var<Poisson::Tags::Field>>(vars_rnd_southwest)) = DataVector{
        0.6964691855978616, 0.28613933495037946, 0.2268514535642031,
        0.5513147690828912, 0.7194689697855631,  0.42310646012446096};
    Vars vars_rnd_northeast{6};
    get(get<Var<Poisson::Tags::Field>>(vars_rnd_northeast)) =
        DataVector{0.5315513738418384, 0.5318275870968661, 0.6344009585513211,
                   0.8494317940777896, 0.7244553248606352, 0.6110235106775829};
    AuxVars expected_aux_vars_rnd{6};
    get<0>(get<Var<::Tags::deriv<Poisson::Tags::Field, tmpl::size_t<2>,
                                 Frame::Inertial>>>(expected_aux_vars_rnd)) =
        DataVector{5.200679648008939,    -0.49983229690025466,
                   -0.16390063442932323, 1.8200149118018878,
                   0.3369321891898911,   1.5677008358240414};
    get<1>(get<Var<::Tags::deriv<Poisson::Tags::Field, tmpl::size_t<2>,
                                 Frame::Inertial>>>(expected_aux_vars_rnd)) =
        DataVector{-0.31839450177748185, -0.7525819072693875,
                   0.6118864945191613,   -2.7457634331575322,
                   -2.0560155094714654,  -2.419963217736805};
    OperatorVars expected_operator_vars_rnd{6};
    get(get<DgOperatorAppliedTo<Var<Poisson::Tags::Field>>>(
        expected_operator_vars_rnd)) =
        DataVector{203.56354715945108, 9.40868981828554,  -2.818657740285368,
                   111.70107437132107, 35.80427083086546, 65.53029015630551};
    test_dg_operator<system, true>(
        domain_creator,
        {{{{northwest_id, std::move(vars_rnd_northwest)},
           {southwest_id, std::move(vars_rnd_southwest)},
           {northeast_id, std::move(vars_rnd_northeast)}},
          {{northwest_id, std::move(expected_aux_vars_rnd)}},
          {{northwest_id, std::move(expected_operator_vars_rnd)}}}});
  }
  {
    INFO("3D");
    const domain::creators::Brick domain_creator{{{-0.5, 0., -1.}},
                                                 {{1.5, 1., 3.}},
                                                 {{false, false, false}},
                                                 {{1, 1, 1}},
                                                 {{2, 3, 4}}};
    using system =
        Poisson::FirstOrderSystem<3, Poisson::Geometry::FlatCartesian>;
    using Vars = Variables<tmpl::list<Var<Poisson::Tags::Field>>>;
    using AuxVars = Variables<tmpl::list<Var<::Tags::deriv<
        Poisson::Tags::Field, tmpl::size_t<3>, Frame::Inertial>>>>;
    using OperatorVars =
        Variables<tmpl::list<DgOperatorAppliedTo<Var<Poisson::Tags::Field>>>>;
    test_dg_operator<system, true>(domain_creator, {});
  }
}
