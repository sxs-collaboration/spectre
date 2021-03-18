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
#include "Evolution/DgSubcell/Actions/TciAndSwitchToDg.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
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

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using variables_tag = Tags::Variables<tmpl::list<Var1>>;
};

template <size_t Dim, typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using initial_tags = tmpl::list<
      ::Tags::TimeStepId, domain::Tags::Mesh<Dim>,
      evolution::dg::subcell::Tags::Mesh<Dim>,
      evolution::dg::subcell::Tags::ActiveGrid,
      evolution::dg::subcell::Tags::DidRollback,
      evolution::dg::subcell::Tags::NeighborDataForReconstructionAndRdmpTci<
          Dim>,
      evolution::dg::subcell::Tags::TciGridHistory,
      Tags::Variables<tmpl::list<Var1>>,
      evolution::dg::subcell::Tags::Inactive<Tags::Variables<tmpl::list<Var1>>>,
      Tags::HistoryEvolvedVariables<Tags::Variables<tmpl::list<Var1>>>,
      Tags::TimeStepper<TimeStepper>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<initial_tags>,
                 evolution::dg::subcell::Actions::TciAndSwitchToDg<
                     typename Metavariables::TciOnSubcellGrid>>>>;
};

template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using component_list = tmpl::list<component<Dim, Metavariables>>;
  using system = System<Dim>;
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

  struct TciOnSubcellGrid {
    using return_tags = tmpl::list<>;
    using argument_tags =
        tmpl::list<evolution::dg::subcell::Tags::Inactive<
                       Tags::Variables<tmpl::list<Var1>>>,
                   Tags::Variables<tmpl::list<Var1>>, domain::Tags::Mesh<Dim>>;

    static bool apply(
        const Variables<
            tmpl::list<evolution::dg::subcell::Tags::Inactive<Var1>>>& dg_vars,
        const Variables<tmpl::list<Var1>>& subcell_vars,
        const Mesh<Dim>& dg_mesh, const double persson_exponent) noexcept {
      Variables<tmpl::list<evolution::dg::subcell::Tags::Inactive<Var1>>>
          reconstructed_dg_vars{dg_vars.number_of_grid_points()};
      evolution::dg::subcell::fd::reconstruct(
          make_not_null(&reconstructed_dg_vars), subcell_vars, dg_mesh,
          evolution::dg::subcell::fd::mesh(dg_mesh).extents());
      CHECK(reconstructed_dg_vars == dg_vars);
      CHECK(approx(persson_exponent) == 5.0);  // Should be subcell_opts + 1
      tci_invoked = true;
      return tci_fails;
    }
  };
};

template <size_t Dim>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool Metavariables<Dim>::rdmp_fails = false;
template <size_t Dim>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool Metavariables<Dim>::tci_fails = false;
template <size_t Dim>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool Metavariables<Dim>::tci_invoked = false;

std::unique_ptr<TimeStepper> make_time_stepper(
    const bool multistep_time_stepper) {
  if (multistep_time_stepper) {
    return std::make_unique<TimeSteppers::AdamsBashforthN>(2);
  } else {
    return std::make_unique<TimeSteppers::RungeKutta3>();
  }
}

template <size_t Dim>
void test_impl(const bool multistep_time_stepper, const bool rdmp_fails,
               const bool tci_fails, const bool always_use_subcell,
               const bool self_starting, const bool in_substep) {
  CAPTURE(Dim);
  CAPTURE(multistep_time_stepper);
  CAPTURE(rdmp_fails);
  CAPTURE(tci_fails);
  CAPTURE(always_use_subcell);
  CAPTURE(self_starting);
  CAPTURE(in_substep);
  if (in_substep and multistep_time_stepper) {
    ERROR("Can't both be taking a substep and using a multistep time stepper");
  }

  using metavars = Metavariables<Dim>;
  metavars::rdmp_fails = rdmp_fails;
  metavars::tci_fails = tci_fails;
  metavars::tci_invoked = false;

  using comp = component<Dim, metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{{evolution::dg::subcell::SubcellOptions{
      1.0e-3, 1.0e-4, 2.0e-3, 2.0e-4, 4.0, 4.0, always_use_subcell}}};

  TimeStepId time_step_id{true, self_starting ? -1 : 1,
                          Time{Slab{1.0, 2.0}, {0, 10}}};
  if (in_substep) {
    // We are in the middle of a time step with a substep method, so update
    // time_step_id to signal it is in a substep.
    time_step_id = TimeStepId{true, 1, Time{Slab{1.0, 2.0}, {0, 10}}, 1,
                              Time{Slab{1.0, 2.0}, {1, 10}}};
  }
  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const evolution::dg::subcell::ActiveGrid active_grid =
      evolution::dg::subcell::ActiveGrid::Subcell;
  const std::unique_ptr<TimeStepper> time_stepper =
      make_time_stepper(multistep_time_stepper);

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
  std::deque<evolution::dg::subcell::ActiveGrid> tci_grid_history{};
  for (size_t i = 0; i < time_stepper->order(); ++i) {
    tci_grid_history.push_back(evolution::dg::subcell::ActiveGrid::Dg);
  }

  using evolved_vars_tags = tmpl::list<Var1>;
  Variables<evolved_vars_tags> evolved_vars{
      subcell_mesh.number_of_grid_points()};
  // Set Var1 to the logical coords, since those are linear
  get(get<Var1>(evolved_vars)) = get<0>(logical_coordinates(subcell_mesh));
  if (rdmp_fails) {
    get(get<Var1>(evolved_vars))[0] = 100.0;
  }
  const Variables<tmpl::list<evolution::dg::subcell::Tags::Inactive<Var1>>>
      inactive_evolved_vars{dg_mesh.number_of_grid_points(), 1.0};
  TimeSteppers::History<
      Variables<evolved_vars_tags>,
      Variables<db::wrap_tags_in<Tags::dt, evolved_vars_tags>>>
      time_stepper_history{};
  for (size_t i = 0; i < time_stepper->order(); ++i) {
    Variables<evolved_vars_tags> vars{subcell_mesh.number_of_grid_points()};
    get(get<Var1>(vars)) =
        (i + 2.0) * get<0>(logical_coordinates(subcell_mesh));
    Variables<db::wrap_tags_in<Tags::dt, evolved_vars_tags>> dt_vars{
        subcell_mesh.number_of_grid_points()};
    get(get<Tags::dt<Var1>>(dt_vars)) =
        (i + 20.0) * get<0>(logical_coordinates(subcell_mesh));
    time_stepper_history.insert(
        {true, 1, Time{Slab{1.0, 2.0}, {static_cast<int>(5 - i), 10}}}, vars,
        dt_vars);
  }

  ActionTesting::emplace_array_component_and_initialize<comp>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0,
      {time_step_id, dg_mesh, subcell_mesh, active_grid, true, neighbor_data,
       tci_grid_history, evolved_vars, inactive_evolved_vars,
       time_stepper_history, make_time_stepper(multistep_time_stepper)});

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
  const auto& tci_grid_history_from_box = ActionTesting::get_databox_tag<
      comp, evolution::dg::subcell::Tags::TciGridHistory>(runner, 0);
  const auto& neighbor_data_from_box = ActionTesting::get_databox_tag<
      comp, evolution::dg::subcell::Tags::
                NeighborDataForReconstructionAndRdmpTci<Dim>>(runner, 0);

  // true if the TCI wasn't invoked at all because we are always using subcell,
  // doing self-start, or took a substep.
  const bool avoid_tci =
      always_use_subcell or self_starting or time_step_id.substep() != 0;

  CHECK_FALSE(ActionTesting::get_databox_tag<
              comp, evolution::dg::subcell::Tags::DidRollback>(runner, 0));

  if (rdmp_fails or avoid_tci) {
    // If the RDMP decided the cell is troubled, we shouldn't be checking the
    // user-specified TCI
    CHECK_FALSE(metavars::tci_invoked);
  }

  // Check ActiveGrid
  if (avoid_tci or rdmp_fails or tci_fails) {
    CHECK(active_grid_from_box == evolution::dg::subcell::ActiveGrid::Subcell);

  } else {
    CHECK(active_grid_from_box == evolution::dg::subcell::ActiveGrid::Dg);
  }

  if (avoid_tci) {
    // Should not have reconstructed DG variables
    CHECK(inactive_vars_from_box == inactive_evolved_vars);
  } else {
    auto reconstructed_dg_vars = inactive_evolved_vars;
    evolution::dg::subcell::fd::reconstruct(
        make_not_null(&reconstructed_dg_vars), evolved_vars, dg_mesh,
        subcell_mesh.extents());
    if (active_grid_from_box == evolution::dg::subcell::ActiveGrid::Subcell) {
      CHECK(reconstructed_dg_vars == inactive_vars_from_box);
    } else {
      // Do swap because types are different
      auto reconstructed_active_vars = evolved_vars;
      swap(reconstructed_dg_vars, reconstructed_active_vars);
      CHECK(reconstructed_active_vars == active_vars_from_box);
    }
  }

  if (active_grid_from_box == evolution::dg::subcell::ActiveGrid::Dg) {
    for (auto expected_it = time_stepper_history.cbegin(),
              box_it = time_stepper_history_from_box.cbegin();
         expected_it != time_stepper_history.end(); ++expected_it, ++box_it) {
      CHECK(expected_it.time_step_id() == box_it.time_step_id());
      CHECK(evolution::dg::subcell::fd::reconstruct(
                expected_it.value(), dg_mesh, subcell_mesh.extents()) ==
            box_it.value());
      CHECK(evolution::dg::subcell::fd::reconstruct(
                expected_it.derivative(), dg_mesh, subcell_mesh.extents()) ==
            box_it.derivative());
    }
    CHECK(neighbor_data_from_box.empty());
    CHECK(tci_grid_history_from_box.empty());
  } else {
    // TCI failed
    for (auto expected_it = time_stepper_history.cbegin(),
              box_it = time_stepper_history_from_box.cbegin();
         expected_it != time_stepper_history.end(); ++expected_it, ++box_it) {
      CHECK(expected_it.time_step_id() == box_it.time_step_id());
      CHECK(expected_it.value() == box_it.value());
      CHECK(expected_it.derivative() == box_it.derivative());
    }
    CHECK(neighbor_data_from_box.size() == 1);
    CHECK(neighbor_data_from_box.count(self_id) == 1);
    if (avoid_tci) {
      CHECK(tci_grid_history_from_box.front() ==
            evolution::dg::subcell::ActiveGrid::Dg);
    } else if (multistep_time_stepper) {
      CHECK(tci_grid_history_from_box.front() ==
            evolution::dg::subcell::ActiveGrid::Subcell);
    } else {
      // substep time steppers don't need to keep track of the history right now
      // because we restrict subcell to DG changes only on step boundaries.
      CHECK(tci_grid_history_from_box.front() ==
            evolution::dg::subcell::ActiveGrid::Dg);
    }
    if (multistep_time_stepper) {
      CHECK(tci_grid_history_from_box.size() == time_stepper->order());
    }
  }
}

template <size_t Dim>
void test() {
  for (const bool use_multistep_time_stepper : {true, false}) {
    for (const bool rdmp_fails : {true, false}) {
      for (const bool tci_fails : {false, true}) {
        for (const bool always_use_subcell : {false, true}) {
          for (const bool self_starting : {false, true}) {
            test_impl<Dim>(use_multistep_time_stepper, rdmp_fails, tci_fails,
                           always_use_subcell, self_starting, false);
            if (not use_multistep_time_stepper) {
              test_impl<Dim>(use_multistep_time_stepper, rdmp_fails, tci_fails,
                             always_use_subcell, self_starting, true);
            }
          }
        }
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Actions.TciAndSwitchToDg",
                  "[Evolution][Unit]") {
  // 1. check that if we are in self-start nothing happens. This can be done by
  //    verifying that the Inactive vars are untouched.
  // 2. check if substep != 0, nothing happens. Check Inactive vars untouched.
  // 3. Check if always_use_subcells, inactive vars are untouched.
  // 4. Check if RDMP gets triggered, then tci_mutator is not called, and we
  //    stay on subcell
  // 5. Check if RDMP is not triggered, but tci_mutator is, we stay on subcell
  // 6. check if RDMP & TCI not triggered, switch to DG.
  Parallel::register_derived_classes_with_charm<TimeStepper>();
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
