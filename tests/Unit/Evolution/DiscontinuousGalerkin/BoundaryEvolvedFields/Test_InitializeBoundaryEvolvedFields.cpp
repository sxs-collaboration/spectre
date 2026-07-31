// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <type_traits>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryEvolvedFields/InitializeBoundaryEvolvedFields.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryEvolvedFields/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// Interior source fields whose boundary twins we evolve.
struct Psi : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct Pi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

using evolution::dg::Tags::BoundaryValue;

struct MockSystem {
  using variables_tag = ::Tags::Variables<tmpl::list<Psi, Pi>>;
};

// The volume history, keyed on the system's evolved variables as in
// production. The initializer reads its `integration_order()` to seed each
// face history; it needs no records.
using volume_history_tag =
    ::Tags::HistoryEvolvedVariables<typename MockSystem::variables_tag>;

// A boundary condition that opts in with two fields, so the projection of
// every union slot is checked.
template <size_t Dim>
class MockOptingBc : public domain::BoundaryConditions::BoundaryCondition {
 public:
  MockOptingBc() = default;
  MockOptingBc(MockOptingBc&&) = default;
  MockOptingBc& operator=(MockOptingBc&&) = default;
  MockOptingBc(const MockOptingBc&) = default;
  MockOptingBc& operator=(const MockOptingBc&) = default;
  ~MockOptingBc() override = default;

  explicit MockOptingBc(CkMigrateMessage* msg)
      : domain::BoundaryConditions::BoundaryCondition(msg) {}

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, MockOptingBc);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override {
    return std::make_unique<MockOptingBc<Dim>>(*this);
  }

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override {
    domain::BoundaryConditions::BoundaryCondition::pup(p);
  }

  using boundary_evolved_variables =
      tmpl::list<BoundaryValue<Psi>, BoundaryValue<Pi>>;
};

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID MockOptingBc<Dim>::my_PUP_ID = 0;

template <size_t Dim>
class MockNonOptingBc : public domain::BoundaryConditions::BoundaryCondition {
 public:
  MockNonOptingBc() = default;
  MockNonOptingBc(MockNonOptingBc&&) = default;
  MockNonOptingBc& operator=(MockNonOptingBc&&) = default;
  MockNonOptingBc(const MockNonOptingBc&) = default;
  MockNonOptingBc& operator=(const MockNonOptingBc&) = default;
  ~MockNonOptingBc() override = default;

  explicit MockNonOptingBc(CkMigrateMessage* msg)
      : domain::BoundaryConditions::BoundaryCondition(msg) {}

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, MockNonOptingBc);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override {
    return std::make_unique<MockNonOptingBc<Dim>>(*this);
  }

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override {
    domain::BoundaryConditions::BoundaryCondition::pup(p);
  }
};

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID MockNonOptingBc<Dim>::my_PUP_ID = 0;

using mock_field_tags = tmpl::list<BoundaryValue<Psi>, BoundaryValue<Pi>>;

template <typename Metavariables>
struct MockComponent {
  static constexpr size_t Dim = Metavariables::volume_dim;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using simple_tags =
      tmpl::list<domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
                 typename MockSystem::variables_tag, volume_history_tag>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, tmpl::list<>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<
              evolution::dg::BoundaryEvolvedFields::
                  InitializeBoundaryEvolvedFields<
                      Dim, MockSystem,
                      tmpl::list<MockOptingBc<Dim>, MockNonOptingBc<Dim>>>>>>;
};
struct MockMetavars {
  static constexpr size_t volume_dim = 2;
  using component_list = tmpl::list<MockComponent<MockMetavars>>;
};

void test_initialize_boundary_evolved_fields(
    const Spectral::Quadrature quadrature) {
  INFO("InitializeBoundaryEvolvedFields on a real element");
  CAPTURE(quadrature);
  constexpr size_t Dim = 2;
  using component = MockComponent<MockMetavars>;
  using values_tag =
      evolution::dg::Tags::BoundaryEvolvedFieldsValues<Dim, mock_field_tags>;
  using dt_stash_tag =
      evolution::dg::Tags::BoundaryEvolvedFieldsDtStash<Dim, mock_field_tags>;
  using history_tag =
      evolution::dg::Tags::BoundaryEvolvedFieldsHistory<Dim, mock_field_tags>;

  // The AMR guard's metavariables detector: a nested `amr` type is the
  // AMR-enable signal.
  struct MetavarsWithAmr {
    struct amr {};
  };
  static_assert(
      evolution::dg::BoundaryEvolvedFields::detail::has_amr_v<MetavarsWithAmr>);
  static_assert(not evolution::dg::BoundaryEvolvedFields::detail::has_amr_v<
                MockMetavars>);

  // The mutator's stored tags are exactly the value, dt-stash, and history
  // maps of the (homogeneous) boundary-evolved field-tag union.
  using mutator =
      evolution::dg::BoundaryEvolvedFields::InitializeBoundaryEvolvedFields<
          Dim, MockSystem, tmpl::list<MockOptingBc<Dim>, MockNonOptingBc<Dim>>>;
  static_assert(
      std::is_same_v<typename mutator::simple_tags,
                     tmpl::list<values_tag, dt_stash_tag, history_tag>>,
      "The init mutator must store the value, dt-stash, and history maps of "
      "the boundary-evolved field-tag union.");

  const Mesh<Dim> mesh{{{3, 4}}, Spectral::Basis::Legendre, quadrature};

  // The element spans its block in xi, so both xi faces and lower_eta are
  // external; upper_eta has a neighbor, so it is interior. The two OPTING
  // faces (lower_xi, lower_eta) slice different dimensions and have different
  // face sizes on the mixed-extent mesh, so the per-face projection and
  // allocation are checked through both sliced dimensions; upper_xi is the
  // non-opting external face.
  const ElementId<Dim> self_id{0, {{{0, 0}, {1, 0}}}};
  const ElementId<Dim> neighbor_eta_id{0, {{{0, 0}, {1, 1}}}};
  const OrientationMap<Dim> orientation = OrientationMap<Dim>::create_aligned();
  typename Element<Dim>::Neighbors_t neighbors{};
  neighbors[Direction<Dim>::upper_eta()] =
      Neighbors<Dim>{{neighbor_eta_id}, orientation};
  const Element<Dim> element{self_id, neighbors};
  const std::array<Direction<Dim>, 2> opting_directions{
      Direction<Dim>::lower_xi(), Direction<Dim>::lower_eta()};
  const auto non_opting_direction = Direction<Dim>::upper_xi();

  // Spatially-varying volume data, distinct per field, so the per-node
  // projection check is meaningful for every union slot.
  Variables<tmpl::list<Psi, Pi>> volume_vars{mesh.number_of_grid_points()};
  for (size_t i = 0; i < mesh.number_of_grid_points(); ++i) {
    get(get<Psi>(volume_vars))[i] = 1.0 + 0.75 * static_cast<double>(i);
    get(get<Pi>(volume_vars))[i] =
        -2.0 + 0.5 * static_cast<double>(i) * static_cast<double>(i);
  }
  const size_t volume_integration_order = 3;

  std::vector<DirectionMap<
      Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      external_boundary_conditions{1};
  for (const auto& direction : opting_directions) {
    external_boundary_conditions[0][direction] =
        std::make_unique<MockOptingBc<Dim>>();
  }
  external_boundary_conditions[0][non_opting_direction] =
      std::make_unique<MockNonOptingBc<Dim>>();

  ActionTesting::MockRuntimeSystem<MockMetavars> runner{
      {std::move(external_boundary_conditions)}};
  ActionTesting::emplace_component_and_initialize<component>(
      make_not_null(&runner), 0,
      {element, mesh, volume_vars,
       typename volume_history_tag::type{volume_integration_order}});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);

  // Face selection: one entry per map per opting external face, nothing else.
  const auto& values =
      ActionTesting::get_databox_tag<component, values_tag>(runner, 0);
  const auto& dt_stash =
      ActionTesting::get_databox_tag<component, dt_stash_tag>(runner, 0);
  const auto& histories =
      ActionTesting::get_databox_tag<component, history_tag>(runner, 0);
  CHECK(values.size() == 2);
  CHECK(dt_stash.size() == 2);
  CHECK(histories.size() == 2);
  CHECK(not values.contains(non_opting_direction));
  CHECK(not values.contains(Direction<Dim>::upper_eta()));

  for (const auto& direction : opting_directions) {
    CAPTURE(direction);
    REQUIRE(values.contains(direction));
    REQUIRE(dt_stash.contains(direction));
    REQUIRE(histories.contains(direction));
    const size_t num_face_pts =
        mesh.slice_away(direction.dimension()).number_of_grid_points();

    // Every union slot is the volume source projected to this face.
    const auto& face_values = values.at(direction);
    REQUIRE(face_values.number_of_grid_points() == num_face_pts);
    tmpl::for_each<tmpl::list<Psi, Pi>>([&face_values, &mesh, &direction,
                                         &num_face_pts,
                                         &volume_vars](auto source_tag_v) {
      using source_tag = tmpl::type_from<decltype(source_tag_v)>;
      Scalar<DataVector> expected{num_face_pts};
      dg::project_tensor_to_boundary(make_not_null(&expected),
                                     get<source_tag>(volume_vars), mesh,
                                     direction);
      CHECK(get(get<BoundaryValue<source_tag>>(face_values)) == get(expected));
    });

    // The dt-stash entry is allocated at face size; the history is empty and
    // seeded at the volume history's integration order.
    CHECK(dt_stash.at(direction).number_of_grid_points() == num_face_pts);
    CHECK(histories.at(direction).integration_order() ==
          volume_integration_order);
    CHECK(histories.at(direction).size() == 0);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Dg.BoundaryEvolvedFields.Initialize",
                  "[Unit][Evolution]") {
  register_classes_with_charm<MockNonOptingBc<2>, MockOptingBc<2>>();

  test_initialize_boundary_evolved_fields(Spectral::Quadrature::Gauss);
  test_initialize_boundary_evolved_fields(Spectral::Quadrature::GaussLobatto);
}
