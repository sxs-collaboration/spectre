// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <deque>
#include <memory>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Actions/TciAndSwitchToDg.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/TciGridHistory.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
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
      evolution::dg::subcell::Tags::NeighborDataForReconstruction<Dim>,
      evolution::dg::subcell::Tags::TciDecision,
      evolution::dg::subcell::Tags::DataForRdmpTci,
      evolution::dg::subcell::Tags::TciGridHistory,
      Tags::Variables<tmpl::list<Var1>>,
      Tags::HistoryEvolvedVariables<Tags::Variables<tmpl::list<Var1>>>,
      Tags::TimeStepper<TimeStepper>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
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

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool rdmp_fails;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool tci_fails;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool tci_invoked;

  struct TciOnSubcellGrid {
    using return_tags = tmpl::list<>;
    using argument_tags =
        tmpl::list<Tags::Variables<tmpl::list<Var1>>,
                   evolution::dg::subcell::Tags::DataForRdmpTci,
                   evolution::dg::subcell::Tags::SubcellOptions>;

    static std::tuple<int, evolution::dg::subcell::RdmpTciData> apply(
        const Variables<tmpl::list<Var1>>& subcell_vars,
        const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,
        const evolution::dg::subcell::SubcellOptions& subcell_options,
        const double persson_exponent) {
      CHECK(approx(persson_exponent) == 5.0);  // Should be subcell_opts + 1
      tci_invoked = true;

      evolution::dg::subcell::RdmpTciData rdmp_data{};
      rdmp_data.max_variables_values =
          std::vector<double>{max(get(get<Var1>(subcell_vars)))};
      rdmp_data.min_variables_values =
          std::vector<double>{min(get(get<Var1>(subcell_vars)))};

      // Now do RDMP check, reconstruct to DG solution, then check.
      CHECK(evolution::dg::subcell::rdmp_tci(
                rdmp_data.max_variables_values, rdmp_data.min_variables_values,
                past_rdmp_tci_data.max_variables_values,
                past_rdmp_tci_data.min_variables_values,
                subcell_options.rdmp_delta0(),
                subcell_options.rdmp_epsilon()) == rdmp_fails);

      return {tci_fails or rdmp_fails, rdmp_data};
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
    return std::make_unique<TimeSteppers::AdamsBashforth>(2);
  } else {
    return std::make_unique<TimeSteppers::Rk3HesthavenSsp>();
  }
}

template <size_t Dim>
void test_impl(
    const bool multistep_time_stepper, const bool rdmp_fails,
    const bool tci_fails, const bool did_rollback,
    const bool always_use_subcell, const bool self_starting,
    const bool in_substep,
    const evolution::dg::subcell::fd::ReconstructionMethod recons_method) {
  CAPTURE(Dim);
  CAPTURE(multistep_time_stepper);
  CAPTURE(rdmp_fails);
  CAPTURE(tci_fails);
  CAPTURE(did_rollback);
  CAPTURE(always_use_subcell);
  CAPTURE(self_starting);
  CAPTURE(in_substep);
  CAPTURE(recons_method);
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
      1.0e-3, 1.0e-4, 2.0e-3, 2.0e-4, 4.0, 4.0, always_use_subcell,
      recons_method}}};

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

  FixedHashMap<maximum_number_of_neighbors(Dim),
               std::pair<Direction<Dim>, ElementId<Dim>>, std::vector<double>,
               boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
      neighbor_data{};

  const int tci_decision{static_cast<int>(tci_fails)};

  evolution::dg::subcell::RdmpTciData rdmp_tci_data{};
  // max and min of +-2 at last time level means reconstructed vars will be in
  // limit
  rdmp_tci_data.max_variables_values.push_back(2.0);
  rdmp_tci_data.min_variables_values.push_back(-2.0);
  std::deque<evolution::dg::subcell::ActiveGrid> tci_grid_history{};
  for (size_t i = 0; i < time_stepper->order(); ++i) {
    tci_grid_history.push_back(evolution::dg::subcell::ActiveGrid::Dg);
  }

  using evolved_vars_tags = tmpl::list<Var1>;
  using dt_evolved_vars_tags = db::wrap_tags_in<Tags::dt, evolved_vars_tags>;
  Variables<evolved_vars_tags> evolved_vars{
      subcell_mesh.number_of_grid_points()};
  // Set Var1 to the logical coords, since those are linear
  get(get<Var1>(evolved_vars)) = get<0>(logical_coordinates(subcell_mesh));
  if (rdmp_fails) {
    get(get<Var1>(evolved_vars))[0] = 100.0;
  }
  TimeSteppers::History<Variables<dt_evolved_vars_tags>> time_stepper_history{};
  for (size_t i = 0; i < time_stepper->order(); ++i) {
    Variables<dt_evolved_vars_tags> dt_vars{
        subcell_mesh.number_of_grid_points()};
    get(get<Tags::dt<Var1>>(dt_vars)) =
        (i + 20.0) * get<0>(logical_coordinates(subcell_mesh));
    time_stepper_history.insert(
        {true, 1, Time{Slab{1.0, 2.0}, {static_cast<int>(5 - i), 10}}},
        dt_vars);
  }
  Variables<evolved_vars_tags> vars{subcell_mesh.number_of_grid_points()};
  get(get<Var1>(vars)) =
      (time_stepper->order() + 1.0) * get<0>(logical_coordinates(subcell_mesh));

  ActionTesting::emplace_array_component_and_initialize<comp>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0,
      {time_step_id, dg_mesh, subcell_mesh, active_grid, did_rollback,
       neighbor_data, tci_decision, rdmp_tci_data, tci_grid_history,
       evolved_vars, time_stepper_history,
       make_time_stepper(multistep_time_stepper)});

  // Invoke the TciAndSwitchToDg action on the runner
  ActionTesting::next_action<comp>(make_not_null(&runner), 0);

  const auto active_grid_from_box =
      ActionTesting::get_databox_tag<comp,
                                     evolution::dg::subcell::Tags::ActiveGrid>(
          runner, 0);
  const auto& active_vars_from_box =
      ActionTesting::get_databox_tag<comp, Tags::Variables<evolved_vars_tags>>(
          runner, 0);
  const auto& time_stepper_history_from_box =
      ActionTesting::get_databox_tag<comp, Tags::HistoryEvolvedVariables<>>(
          runner, 0);
  const auto& tci_grid_history_from_box = ActionTesting::get_databox_tag<
      comp, evolution::dg::subcell::Tags::TciGridHistory>(runner, 0);

  // true if the TCI wasn't invoked at all because we are always using subcell,
  // doing self-start, took a substep, or already did rollback from DG to FD.
  const bool avoid_tci = always_use_subcell or self_starting or
                         time_step_id.substep() != 0 or did_rollback;

  CHECK_FALSE(ActionTesting::get_databox_tag<
              comp, evolution::dg::subcell::Tags::DidRollback>(runner, 0));

  if (avoid_tci) {
    CHECK_FALSE(metavars::tci_invoked);
  }

  // Check ActiveGrid
  if (avoid_tci or rdmp_fails or tci_fails) {
    CHECK(active_grid_from_box == evolution::dg::subcell::ActiveGrid::Subcell);
  } else {
    CHECK(active_grid_from_box == evolution::dg::subcell::ActiveGrid::Dg);
  }

  if (not avoid_tci) {
    Variables<tmpl::list<Var1>> reconstructed_dg_vars{
        dg_mesh.number_of_grid_points()};
    evolution::dg::subcell::fd::reconstruct(
        make_not_null(&reconstructed_dg_vars), evolved_vars, dg_mesh,
        subcell_mesh.extents(), recons_method);
    if (active_grid_from_box == evolution::dg::subcell::ActiveGrid::Dg) {
      // Do swap because types are different
      // auto reconstructed_active_vars = evolved_vars;
      // swap(reconstructed_dg_vars, reconstructed_active_vars);
      CHECK(reconstructed_dg_vars == active_vars_from_box);
    }
  }

  if (active_grid_from_box == evolution::dg::subcell::ActiveGrid::Dg) {
    for (auto expected_it = time_stepper_history.derivatives_begin(),
              box_it = time_stepper_history_from_box.derivatives_begin();
         expected_it != time_stepper_history.derivatives_end();
         ++expected_it, ++box_it) {
      CHECK(expected_it.time_step_id() == box_it.time_step_id());
      CHECK(evolution::dg::subcell::fd::reconstruct(*expected_it, dg_mesh,
                                                    subcell_mesh.extents(),
                                                    recons_method) == *box_it);
    }
    CHECK(tci_grid_history_from_box.empty());
  } else {
    // TCI failed
    for (auto expected_it = time_stepper_history.derivatives_begin(),
              box_it = time_stepper_history_from_box.derivatives_begin();
         expected_it != time_stepper_history.derivatives_end();
         ++expected_it, ++box_it) {
      CHECK(expected_it.time_step_id() == box_it.time_step_id());
      CHECK(*expected_it == *box_it);
    }
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
        for (const bool did_rollback : {true, false}) {
          for (const bool always_use_subcell : {false, true}) {
            for (const bool self_starting : {false, true}) {
              for (const auto recons_method :
                   {evolution::dg::subcell::fd::ReconstructionMethod::
                        AllDimsAtOnce,
                    evolution::dg::subcell::fd::ReconstructionMethod::
                        DimByDim}) {
                test_impl<Dim>(use_multistep_time_stepper, rdmp_fails,
                               tci_fails, did_rollback, always_use_subcell,
                               self_starting, false, recons_method);
                if (not use_multistep_time_stepper) {
                  test_impl<Dim>(use_multistep_time_stepper, rdmp_fails,
                                 tci_fails, did_rollback, always_use_subcell,
                                 self_starting, true, recons_method);
                }
              }
            }
          }
        }
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Actions.TciAndSwitchToDg",
                  "[Evolution][Unit]") {
  // 1. check that if we are in self-start nothing happens.
  // 2. check if substep != 0, nothing happens.
  // 3. Check if always_use_subcells.
  // 4. Check if RDMP gets triggered, then tci_mutator is not called, and we
  //    stay on subcell
  // 5. Check if RDMP is not triggered, but tci_mutator is, we stay on subcell
  // 6. check if RDMP & TCI not triggered, switch to DG.
  // 7. check if DidRollBack=True, stay in subcell.
  Parallel::register_classes_with_charm<TimeSteppers::AdamsBashforth,
                                        TimeSteppers::Rk3HesthavenSsp>();
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
