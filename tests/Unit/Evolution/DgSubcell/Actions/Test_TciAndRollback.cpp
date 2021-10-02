// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <deque>
#include <memory>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Actions/TciAndRollback.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/TciGridHistory.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct PrimVar1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim, bool HasPrims>
struct System {
  static constexpr size_t volume_dim = Dim;
  static constexpr bool has_primitive_and_conservative_vars = HasPrims;
  using variables_tag = Tags::Variables<tmpl::list<Var1>>;
  using primitive_variables_tag = ::Tags::Variables<tmpl::list<PrimVar1>>;
};

template <size_t>
struct DummyLabel;

template <size_t Dim, typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using initial_tags = tmpl::append<
      tmpl::list<
          ::Tags::TimeStepId, domain::Tags::Mesh<Dim>,
          evolution::dg::subcell::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
          evolution::dg::subcell::Tags::ActiveGrid,
          evolution::dg::subcell::Tags::DidRollback,
          evolution::dg::subcell::Tags::NeighborDataForReconstructionAndRdmpTci<
              Dim>,
          Tags::Variables<tmpl::list<Var1>>,
          evolution::dg::subcell::Tags::Inactive<
              Tags::Variables<tmpl::list<Var1>>>,
          Tags::HistoryEvolvedVariables<Tags::Variables<tmpl::list<Var1>>>,
          SelfStart::Tags::InitialValue<Tags::Variables<tmpl::list<Var1>>>>,
      tmpl::conditional_t<
          Metavariables::has_prims,
          tmpl::list<Tags::Variables<tmpl::list<PrimVar1>>,
                     SelfStart::Tags::InitialValue<
                         Tags::Variables<tmpl::list<PrimVar1>>>>,
          tmpl::list<>>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<initial_tags>,
                 evolution::dg::subcell::Actions::TciAndRollback<
                     typename Metavariables::TciOnDgGrid>,
                 Actions::Label<DummyLabel<0>>,
                 Actions::Label<evolution::dg::subcell::Actions::Labels::
                                    BeginSubcellAfterDgRollback>,
                 Actions::Label<DummyLabel<1>>>>>;
};

template <size_t Dim, bool HasPrims>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  static constexpr bool has_prims = HasPrims;
  using component_list = tmpl::list<component<Dim, Metavariables>>;
  using system = System<Dim, HasPrims>;
  using analytic_variables_tags = typename system::variables_tag::tags_list;
  using const_global_cache_tags =
      tmpl::list<evolution::dg::subcell::Tags::SubcellOptions>;
  enum class Phase { Initialization, Exit };

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool rdmp_fails;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool tci_fails;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool tci_invoked;

  struct TciOnDgGrid {
    using return_tags = tmpl::list<>;
    using argument_tags =
        tmpl::list<evolution::dg::subcell::Tags::Inactive<
                       Tags::Variables<tmpl::list<Var1>>>,
                   Tags::Variables<tmpl::list<Var1>>, domain::Tags::Mesh<Dim>>;

    static bool apply(
        const Variables<tmpl::list<
            evolution::dg::subcell::Tags::Inactive<Var1>>>& subcell_vars,
        const Variables<tmpl::list<Var1>>& dg_vars, const Mesh<Dim>& dg_mesh,
        const double persson_exponent) {
      Variables<tmpl::list<evolution::dg::subcell::Tags::Inactive<Var1>>>
          projected_vars{subcell_vars.number_of_grid_points()};
      evolution::dg::subcell::fd::project(
          make_not_null(&projected_vars), dg_vars, dg_mesh,
          evolution::dg::subcell::fd::mesh(dg_mesh).extents());
      CHECK(projected_vars == subcell_vars);
      CHECK(approx(persson_exponent) == 4.0);
      tci_invoked = true;
      return tci_fails;
    }
  };
};

template <size_t Dim, bool HasPrims>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool Metavariables<Dim, HasPrims>::rdmp_fails = false;
template <size_t Dim, bool HasPrims>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool Metavariables<Dim, HasPrims>::tci_fails = false;
template <size_t Dim, bool HasPrims>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool Metavariables<Dim, HasPrims>::tci_invoked = false;

template <size_t Dim>
Element<Dim> create_element(const bool with_neighbors) {
  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  if (with_neighbors) {
    for (size_t i = 0; i < 2 * Dim; ++i) {
      neighbors[gsl::at(Direction<Dim>::all_directions(), i)] =
          Neighbors<Dim>{{ElementId<Dim>{i + 1, {}}}, {}};
    }
  }
  return Element<Dim>{ElementId<Dim>{0, {}}, neighbors};
}

template <size_t Dim, bool HasPrims>
void test_impl(const bool rdmp_fails, const bool tci_fails,
               const bool always_use_subcell, const bool self_starting,
               const bool with_neighbors) {
  CAPTURE(Dim);
  CAPTURE(rdmp_fails);
  CAPTURE(tci_fails);
  CAPTURE(always_use_subcell);
  CAPTURE(self_starting);
  CAPTURE(with_neighbors);

  using metavars = Metavariables<Dim, HasPrims>;
  metavars::rdmp_fails = rdmp_fails;
  metavars::tci_fails = tci_fails;
  metavars::tci_invoked = false;

  using comp = component<Dim, metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{{evolution::dg::subcell::SubcellOptions{
      1.0e-3, 1.0e-4, 2.0e-3, 2.0e-4, 5.0, 4.0, always_use_subcell}}};

  const TimeStepId time_step_id{true, self_starting ? -1 : 1,
                                Time{Slab{1.0, 2.0}, {0, 10}}};
  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const Element<Dim> element = create_element<Dim>(with_neighbors);
  const evolution::dg::subcell::ActiveGrid active_grid =
      evolution::dg::subcell::ActiveGrid::Dg;

  FixedHashMap<maximum_number_of_neighbors(Dim) + 1,
               std::pair<Direction<Dim>, ElementId<Dim>>,
               evolution::dg::subcell::NeighborData,
               boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
      neighbor_data{};
  const std::pair self_id{Direction<Dim>::lower_xi(),
                          ElementId<Dim>::external_boundary_id()};
  neighbor_data[self_id] = {};
  // max and min of +-2 at last time level means reconstructed vars will be in
  // limit
  neighbor_data[self_id].max_variables_values.push_back(2.0);
  neighbor_data[self_id].min_variables_values.push_back(-2.0);

  using evolved_vars_tags = tmpl::list<Var1>;
  Variables<evolved_vars_tags> evolved_vars{dg_mesh.number_of_grid_points()};
  // Set Var1 to the logical coords, since those are linear
  get(get<Var1>(evolved_vars)) = get<0>(logical_coordinates(dg_mesh));
  if (rdmp_fails) {
    get(get<Var1>(evolved_vars))[0] = 100.0;
  }
  using prim_vars_tags = tmpl::list<PrimVar1>;
  Variables<prim_vars_tags> prim_vars{dg_mesh.number_of_grid_points()};
  get(get<PrimVar1>(prim_vars)) = get<0>(logical_coordinates(dg_mesh)) + 1000.0;

  const Variables<tmpl::list<evolution::dg::subcell::Tags::Inactive<Var1>>>
      inactive_evolved_vars{subcell_mesh.number_of_grid_points(), 1.0};
  constexpr size_t history_size = 5;
  TimeSteppers::History<
      Variables<evolved_vars_tags>,
      Variables<db::wrap_tags_in<Tags::dt, evolved_vars_tags>>>
      time_stepper_history{history_size};
  for (size_t i = 0; i < history_size; ++i) {
    Variables<db::wrap_tags_in<Tags::dt, evolved_vars_tags>> dt_vars{
        dg_mesh.number_of_grid_points()};
    get(get<Tags::dt<Var1>>(dt_vars)) =
        (i + 20.0) * get<0>(logical_coordinates(dg_mesh));
    time_stepper_history.insert(
        {true, 1, Time{Slab{1.0, 2.0}, {static_cast<int>(5 - i), 10}}},
        dt_vars);
  }
  Variables<evolved_vars_tags> vars{dg_mesh.number_of_grid_points()};
  get(get<Var1>(vars)) =
      (history_size + 1.0) * get<0>(logical_coordinates(dg_mesh));
  time_stepper_history.most_recent_value() = vars;

  const bool did_rollback = false;
  Variables<evolved_vars_tags> initial_value_evolved_vars{
      dg_mesh.number_of_grid_points(), 1.0e8};
  Variables<prim_vars_tags> initial_value_prim_vars{
      dg_mesh.number_of_grid_points(), 1.0e10};

  if constexpr (HasPrims) {
    ActionTesting::emplace_array_component_and_initialize<comp>(
        &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0,
        {time_step_id, dg_mesh, subcell_mesh, element, active_grid,
         did_rollback, neighbor_data, evolved_vars, inactive_evolved_vars,
         time_stepper_history, initial_value_evolved_vars, prim_vars,
         initial_value_prim_vars});
  } else {
    (void)prim_vars;
    (void)initial_value_prim_vars;
    ActionTesting::emplace_array_component_and_initialize<comp>(
        &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0,
        {time_step_id, dg_mesh, subcell_mesh, element, active_grid,
         did_rollback, neighbor_data, evolved_vars, inactive_evolved_vars,
         time_stepper_history, initial_value_evolved_vars});
  }

  // Invoke the TciAndSwitchToDg action on the runner
  ActionTesting::next_action<comp>(make_not_null(&runner), 0);

  const auto active_grid_from_box =
      ActionTesting::get_databox_tag<comp,
                                     evolution::dg::subcell::Tags::ActiveGrid>(
          runner, 0);
  const auto& inactive_vars_from_box =
      ActionTesting::get_databox_tag<comp,
                                     evolution::dg::subcell::Tags::Inactive<
                                         Tags::Variables<evolved_vars_tags>>>(
          runner, 0);
  const auto& active_vars_from_box =
      ActionTesting::get_databox_tag<comp, Tags::Variables<evolved_vars_tags>>(
          runner, 0);
  const auto& time_stepper_history_from_box =
      ActionTesting::get_databox_tag<comp, Tags::HistoryEvolvedVariables<>>(
          runner, 0);
  const auto& neighbor_data_from_box = ActionTesting::get_databox_tag<
      comp, evolution::dg::subcell::Tags::
                NeighborDataForReconstructionAndRdmpTci<Dim>>(runner, 0);
  const auto& did_rollback_from_box =
      ActionTesting::get_databox_tag<comp,
                                     evolution::dg::subcell::Tags::DidRollback>(
          runner, 0);
  const auto& initial_value_evolved_vars_from_box =
      get<0>(ActionTesting::get_databox_tag<
             comp,
             SelfStart::Tags::InitialValue<Tags::Variables<evolved_vars_tags>>>(
          runner, 0));

  const bool expected_rollback =
      with_neighbors and (always_use_subcell or rdmp_fails or tci_fails);

  if (expected_rollback) {
    CHECK(ActionTesting::get_next_action_index<comp>(runner, 0) == 4);
    CHECK(active_grid_from_box == evolution::dg::subcell::ActiveGrid::Subcell);
    CHECK_FALSE(neighbor_data_from_box.empty());
    CHECK(did_rollback_from_box);

    CHECK(time_stepper_history_from_box.size() == history_size - 1);
    CHECK(time_stepper_history_from_box.integration_order() ==
          time_stepper_history.integration_order());
    TimeSteppers::History<
        Variables<evolved_vars_tags>,
        Variables<db::wrap_tags_in<Tags::dt, evolved_vars_tags>>>
        time_stepper_history_subcells{time_stepper_history.integration_order()};
    time_stepper_history_subcells.most_recent_value() =
        evolution::dg::subcell::fd::project(
            time_stepper_history.most_recent_value(), dg_mesh,
            subcell_mesh.extents());
    const auto end_it = std::prev(time_stepper_history.end());
    for (auto it = time_stepper_history.begin(); it != end_it; ++it) {
      time_stepper_history_subcells.insert(
          it.time_step_id(),
          evolution::dg::subcell::fd::project(it.derivative(), dg_mesh,
                                              subcell_mesh.extents()));
    }
    REQUIRE(time_stepper_history_subcells.size() ==
            time_stepper_history_from_box.size());
    for (auto expected_it = time_stepper_history_subcells.begin(),
              it = time_stepper_history_from_box.begin();
         expected_it != time_stepper_history_subcells.end();
         ++it, ++expected_it) {
      CHECK(it.time_step_id() == expected_it.time_step_id());
      CHECK(it.derivative() == expected_it.derivative());
    }

    CHECK(get<evolution::dg::subcell::Tags::Inactive<Var1>>(
              inactive_vars_from_box) ==
          get<Var1>(time_stepper_history.most_recent_value()));
    CHECK(active_vars_from_box ==
          evolution::dg::subcell::fd::project(
              time_stepper_history.most_recent_value(), dg_mesh,
              subcell_mesh.extents()));

    if (self_starting) {
      CHECK(initial_value_evolved_vars_from_box ==
            evolution::dg::subcell::fd::project(
                initial_value_evolved_vars, dg_mesh, subcell_mesh.extents()));
      if constexpr (HasPrims) {
        const auto& initial_value_prim_vars_from_box = get<0>(
            ActionTesting::get_databox_tag<
                comp,
                SelfStart::Tags::InitialValue<Tags::Variables<prim_vars_tags>>>(
                runner, 0));
        CHECK(initial_value_prim_vars_from_box ==
              evolution::dg::subcell::fd::project(
                  initial_value_prim_vars, dg_mesh, subcell_mesh.extents()));
      }
    }
  } else {
    CHECK(ActionTesting::get_next_action_index<comp>(runner, 0) == 2);
    CHECK(active_grid_from_box == evolution::dg::subcell::ActiveGrid::Dg);
    CHECK(neighbor_data_from_box.empty());
    CHECK_FALSE(did_rollback_from_box);

    CHECK(time_stepper_history_from_box.size() == history_size);
    CHECK(time_stepper_history_from_box.integration_order() ==
          time_stepper_history.integration_order());
    CHECK(time_stepper_history_from_box.most_recent_value() ==
          time_stepper_history.most_recent_value());
    for (auto expected_it = time_stepper_history.begin(),
              it = time_stepper_history_from_box.begin();
         expected_it != time_stepper_history.end(); ++it, ++expected_it) {
      CHECK(it.time_step_id() == expected_it.time_step_id());
      CHECK(it.derivative() == expected_it.derivative());
    }

    if (with_neighbors) {
      CHECK(get(get<evolution::dg::subcell::Tags::Inactive<Var1>>(
                inactive_vars_from_box)) ==
            evolution::dg::subcell::fd::project(
                get(get<Var1>(evolved_vars)), dg_mesh, subcell_mesh.extents()));
    }

    CHECK(active_vars_from_box == evolved_vars);
    if (self_starting) {
      CHECK(initial_value_evolved_vars_from_box == initial_value_evolved_vars);
      if constexpr (HasPrims) {
        const auto& initial_value_prim_vars_from_box = get<0>(
            ActionTesting::get_databox_tag<
                comp,
                SelfStart::Tags::InitialValue<Tags::Variables<prim_vars_tags>>>(
                runner, 0));
        CHECK(initial_value_prim_vars_from_box == initial_value_prim_vars);
      }
    }
  }
}

template <size_t Dim>
void test() {
  for (const bool rdmp_fails : {true, false}) {
    for (const bool tci_fails : {false, true}) {
      for (const bool always_use_subcell : {false, true}) {
        for (const bool self_starting : {false, true}) {
          for (const bool have_neighbors : {false, true}) {
            test_impl<Dim, true>(rdmp_fails, tci_fails, always_use_subcell,
                                 self_starting, have_neighbors);
            test_impl<Dim, false>(rdmp_fails, tci_fails, always_use_subcell,
                                  self_starting, have_neighbors);
          }
        }
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Actions.TciAndRollback",
                  "[Evolution][Unit]") {
  // We test the following cases:
  // 1. Test RDMP passes/fails (check TciMutator not called on failure)
  // 2. Test always_use_subcells
  // 3. Test TciMutator passes/fails
  //
  // Below is a list of quantities to verify were handled/set correctly by the
  // action:
  // - history projected with correct size (one fewer)
  // - active vars become projection of latest in history
  // - active_grid is correct
  // - did_rollback is correct
  // - if self-start check initial value (and prims) were projected
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
