// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/functional/hash.hpp>
#include <catch.hpp>
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
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Actions/TciAndSwitchToDg.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/TciGridHistory.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Phase.hpp"
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
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
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
  using flux_variables = tmpl::list<Var1>;
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
      evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>,
      evolution::dg::subcell::Tags::TciDecision,
      evolution::dg::subcell::Tags::DataForRdmpTci,
      evolution::dg::subcell::Tags::TciGridHistory,
      Tags::Variables<tmpl::list<Var1>>,
      Tags::HistoryEvolvedVariables<Tags::Variables<tmpl::list<Var1>>>,
      Tags::TimeStepper<TimeStepper>,
      evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>,
      domain::Tags::Element<Dim>,
      evolution::dg::subcell::Tags::CellCenteredFlux<
          typename metavariables::system::flux_variables, Dim>>;

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
      tmpl::list<evolution::dg::subcell::Tags::SubcellOptions<Dim>>;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool rdmp_fails;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool tci_fails;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool tci_invoked;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool tci_rdmp_data_only;

  struct TciOnSubcellGrid {
    using return_tags = tmpl::list<>;
    using argument_tags =
        tmpl::list<Tags::Variables<tmpl::list<Var1>>,
                   evolution::dg::subcell::Tags::DataForRdmpTci,
                   evolution::dg::subcell::Tags::SubcellOptions<Dim>>;

    static std::tuple<int, evolution::dg::subcell::RdmpTciData> apply(
        const Variables<tmpl::list<Var1>>& subcell_vars,
        const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,
        const evolution::dg::subcell::SubcellOptions& subcell_options,
        const double persson_exponent, const bool need_rdmp_data_only) {
      CHECK(approx(persson_exponent) == 5.0);  // Should be subcell_opts + 1
      tci_invoked = true;
      tci_rdmp_data_only = need_rdmp_data_only;

      evolution::dg::subcell::RdmpTciData rdmp_data{};
      rdmp_data.max_variables_values =
          DataVector{max(get(get<Var1>(subcell_vars)))};
      rdmp_data.min_variables_values =
          DataVector{min(get(get<Var1>(subcell_vars)))};

      // Now do RDMP check, reconstruct to DG solution, then check.
      CHECK(evolution::dg::subcell::rdmp_tci(
                rdmp_data.max_variables_values, rdmp_data.min_variables_values,
                past_rdmp_tci_data.max_variables_values,
                past_rdmp_tci_data.min_variables_values,
                subcell_options.rdmp_delta0(),
                subcell_options.rdmp_epsilon()) == rdmp_fails);

      const int decision = rdmp_fails ? -10 : (tci_fails ? -5 : 0);
      return {decision, rdmp_data};
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
template <size_t Dim>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool Metavariables<Dim>::tci_rdmp_data_only = false;

std::unique_ptr<TimeStepper> make_time_stepper(
    const bool multistep_time_stepper) {
  if (multistep_time_stepper) {
    return std::make_unique<TimeSteppers::AdamsBashforth>(2);
  } else {
    return std::make_unique<TimeSteppers::Rk3HesthavenSsp>();
  }
}

template <size_t Dim>
class TestCreator : public DomainCreator<Dim> {
  Domain<Dim> create_domain() const override { return Domain<Dim>{}; }
  std::vector<DirectionMap<
      Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override {
    return {};
  }

  std::vector<std::string> block_names() const override { return {"Block0"}; }

  std::vector<std::array<size_t, Dim>> initial_extents() const override {
    return {};
  }

  std::vector<std::array<size_t, Dim>> initial_refinement_levels()
      const override {
    return {};
  }
};

template <size_t Dim>
void test_impl(
    const bool multistep_time_stepper, const bool rdmp_fails,
    const bool tci_fails, const bool did_rollback,
    const bool always_use_subcell, const bool self_starting,
    const bool in_substep,
    const evolution::dg::subcell::fd::ReconstructionMethod recons_method,
    const bool use_halo, const bool neighbor_is_troubled,
    const bool test_block_id_assert) {
  CAPTURE(Dim);
  CAPTURE(multistep_time_stepper);
  CAPTURE(rdmp_fails);
  CAPTURE(tci_fails);
  CAPTURE(did_rollback);
  CAPTURE(always_use_subcell);
  CAPTURE(self_starting);
  CAPTURE(in_substep);
  CAPTURE(recons_method);
  CAPTURE(use_halo);
  CAPTURE(neighbor_is_troubled);
  CAPTURE(test_block_id_assert);
  if (in_substep and multistep_time_stepper) {
    ERROR("Can't both be taking a substep and using a multistep time stepper");
  }

  using metavars = Metavariables<Dim>;
  metavars::rdmp_fails = rdmp_fails;
  metavars::tci_fails = tci_fails;
  metavars::tci_invoked = false;
  metavars::tci_rdmp_data_only = true;

  using comp = component<Dim, metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{{evolution::dg::subcell::SubcellOptions{
      evolution::dg::subcell::SubcellOptions{
          1.0e-3, 1.0e-4, 2.0e-3, 2.0e-4, 4.0, 4.0, always_use_subcell,
          recons_method, use_halo,
          test_block_id_assert
              ? std::optional{std::vector<std::string>{"Block0"}}
              : std::optional<std::vector<std::string>>{},
          ::fd::DerivativeOrder::Two},
      TestCreator<Dim>{}}}};

  TimeStepId time_step_id{false, self_starting ? -1 : 1,
                          Slab{1.0, 2.0}.end()};
  const TimeDelta step_size{Slab{1.0, 2.0}, {-1, 10}};
  if (in_substep) {
    // We are in the middle of a time step with a substep method, so update
    // time_step_id to signal it is in a substep.
    time_step_id =
        TimeStepId{false, 1, Slab{1.0, 2.0}.end(), 1, step_size, 1.1};
  }
  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const evolution::dg::subcell::ActiveGrid active_grid =
      evolution::dg::subcell::ActiveGrid::Subcell;
  const std::unique_ptr<TimeStepper> time_stepper =
      make_time_stepper(multistep_time_stepper);

  FixedHashMap<maximum_number_of_neighbors(Dim),
               std::pair<Direction<Dim>, ElementId<Dim>>,
               evolution::dg::subcell::GhostData,
               boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
      ghost_data{};

  const int tci_decision{-1};  // default value

  // max and min of +-2 at last time level means reconstructed vars will be in
  // limit
  evolution::dg::subcell::RdmpTciData rdmp_tci_data{{2.0}, {-2.0}};
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
  TimeSteppers::History<Variables<evolved_vars_tags>> time_stepper_history{4};
  {
    constexpr size_t history_size = 5;
    constexpr size_t history_substeps = 3;
    Time step_time{Slab{1.0, 2.0}, {6, 10}};
    for (size_t i = 0; i < history_size; ++i) {
      step_time += step_size;
      Variables<dt_evolved_vars_tags> dt_vars{
          subcell_mesh.number_of_grid_points()};
      get(get<Tags::dt<Var1>>(dt_vars)) =
          (i + 20.0) * get<0>(logical_coordinates(subcell_mesh));
      time_stepper_history.insert({false, 1, step_time}, i * evolved_vars,
                                  dt_vars);
    }
    for (size_t i = 0; i < history_substeps; ++i) {
      Variables<dt_evolved_vars_tags> dt_vars{
          subcell_mesh.number_of_grid_points()};
      get(get<Tags::dt<Var1>>(dt_vars)) =
          (i + 40.0) * get<0>(logical_coordinates(subcell_mesh));
      time_stepper_history.insert(
          {false, 1, step_time, i + 1, step_size, step_time.value()},
          -i * evolved_vars, dt_vars);
    }
    time_stepper_history.discard_value(time_stepper_history[2].time_step_id);
    time_stepper_history.discard_value(
        time_stepper_history.substeps()[1].time_step_id);
  }
  Variables<evolved_vars_tags> vars{subcell_mesh.number_of_grid_points()};
  get(get<Var1>(vars)) =
      (time_stepper->order() + 1.0) * get<0>(logical_coordinates(subcell_mesh));

  typename evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>::type
      neighbor_decisions{};
  neighbor_decisions.insert(
      std::pair{std::pair{Direction<Dim>::lower_xi(), ElementId<Dim>{10}},
                neighbor_is_troubled ? 10 : 0});

  ActionTesting::emplace_array_component_and_initialize<comp>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0,
      {time_step_id, dg_mesh, subcell_mesh, active_grid, did_rollback,
       ghost_data, tci_decision, rdmp_tci_data, tci_grid_history, evolved_vars,
       time_stepper_history, make_time_stepper(multistep_time_stepper),
       neighbor_decisions, Element<Dim>{ElementId<Dim>{0}, {}},
       typename evolution::dg::subcell::Tags::CellCenteredFlux<
           typename metavars::system::flux_variables, Dim>::type::value_type{
           subcell_mesh.number_of_grid_points()}});

  // Invoke the TciAndSwitchToDg action on the runner
  if (test_block_id_assert) {
    CHECK_THROWS_WITH(
        ActionTesting::next_action<comp>(make_not_null(&runner), 0),
        Catch::Matchers::Contains("Should never use subcell on element "));
    return;
  }
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
  const auto& cell_centered_flux_from_box = ActionTesting::get_databox_tag<
      comp, evolution::dg::subcell::Tags::CellCenteredFlux<
                typename metavars::system::flux_variables, Dim>>(runner, 0);

  // true if the TCI wasn't invoked at all because we are always using subcell,
  // doing self-start, took a substep, or already did rollback from DG to FD.
  const bool avoid_tci = always_use_subcell;
  const bool avoid_switch_to_dg =
      avoid_tci or self_starting or time_step_id.substep() != 0 or did_rollback;

  CHECK_FALSE(ActionTesting::get_databox_tag<
              comp, evolution::dg::subcell::Tags::DidRollback>(runner, 0));

  CHECK(metavars::tci_rdmp_data_only == avoid_switch_to_dg);

  if (avoid_tci) {
    CHECK_FALSE(metavars::tci_invoked);
  }

  // Check ActiveGrid
  if (avoid_switch_to_dg or rdmp_fails or tci_fails or
      (use_halo and neighbor_is_troubled)) {
    CHECK(active_grid_from_box == evolution::dg::subcell::ActiveGrid::Subcell);
    CHECK(cell_centered_flux_from_box.has_value());
  } else {
    CHECK(active_grid_from_box == evolution::dg::subcell::ActiveGrid::Dg);
    CHECK(not cell_centered_flux_from_box.has_value());
  }

  if (not avoid_switch_to_dg) {
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
    CHECK(time_stepper_history_from_box.size() == time_stepper_history.size());
    CHECK(time_stepper_history_from_box.substeps().size() ==
          time_stepper_history.substeps().size());
    CHECK(time_stepper_history_from_box.integration_order() ==
          time_stepper_history.integration_order());
    {
      const auto check_box_record = [&](const auto& original_record) {
        const auto& record_from_box =
            time_stepper_history_from_box[original_record.time_step_id];
        CHECK(record_from_box.derivative ==
              evolution::dg::subcell::fd::reconstruct(
                  original_record.derivative, dg_mesh, subcell_mesh.extents(),
                  recons_method));
        if (original_record.value.has_value()) {
          CHECK(record_from_box.value ==
                std::optional{evolution::dg::subcell::fd::reconstruct(
                    *original_record.value, dg_mesh, subcell_mesh.extents(),
                    recons_method)});
        } else {
          CHECK(not record_from_box.value.has_value());
        }
      };
      for (const auto& original_record : time_stepper_history) {
        check_box_record(original_record);
      }
      for (const auto& original_record : time_stepper_history.substeps()) {
        check_box_record(original_record);
      }
    }
    CHECK(tci_grid_history_from_box.empty());
  } else {
    // TCI failed
    CHECK(time_stepper_history_from_box.size() == time_stepper_history.size());
    CHECK(time_stepper_history_from_box.substeps().size() ==
          time_stepper_history.substeps().size());
    CHECK(time_stepper_history_from_box.integration_order() ==
          time_stepper_history.integration_order());
    for (const auto& original_record : time_stepper_history) {
      CHECK(time_stepper_history_from_box[original_record.time_step_id] ==
            original_record);
    }
    for (const auto& original_record : time_stepper_history.substeps()) {
      CHECK(time_stepper_history_from_box[original_record.time_step_id] ==
            original_record);
    }
    if (avoid_switch_to_dg) {
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
  CHECK(ActionTesting::get_databox_tag<
            comp, evolution::dg::subcell::Tags::TciDecision>(runner, 0) ==
        (avoid_switch_to_dg ? -1 : (rdmp_fails ? -10 : (tci_fails ? -5 : 0))));
}

template <size_t Dim>
void test() {
#ifdef SPECTRE_DEBUG
  bool tested_block_id_assert = false;
#endif
  for (const bool use_multistep_time_stepper : {true, false}) {
    for (const bool rdmp_fails : {true, false}) {
      for (const bool tci_fails : {false, true}) {
        for (const bool did_rollback : {true, false}) {
          for (const bool always_use_subcell : {false, true}) {
            for (const bool self_starting : {false, true}) {
              for (const bool use_halo : {false, true}) {
                for (const bool neighbor_is_troubled : {false, true}) {
                  for (const auto recons_method :
                       {evolution::dg::subcell::fd::ReconstructionMethod::
                            AllDimsAtOnce,
                        evolution::dg::subcell::fd::ReconstructionMethod::
                            DimByDim}) {
                    test_impl<Dim>(use_multistep_time_stepper, rdmp_fails,
                                   tci_fails, did_rollback, always_use_subcell,
                                   self_starting, false, recons_method,
                                   use_halo, neighbor_is_troubled, false);
                    if (not use_multistep_time_stepper) {
                      test_impl<Dim>(
                          use_multistep_time_stepper, rdmp_fails, tci_fails,
                          did_rollback, always_use_subcell, self_starting, true,
                          recons_method, use_halo, neighbor_is_troubled, false);
                    }
#ifdef SPECTRE_DEBUG
                    if (not tested_block_id_assert) {
                      test_impl<Dim>(use_multistep_time_stepper, rdmp_fails,
                                     tci_fails, did_rollback,
                                     always_use_subcell, self_starting, false,
                                     recons_method, use_halo,
                                     neighbor_is_troubled, true);
                      tested_block_id_assert = true;
                    }
#endif
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

// [[TimeOut, 10]]
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
  register_classes_with_charm<TimeSteppers::AdamsBashforth,
                              TimeSteppers::Rk3HesthavenSsp>();
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
