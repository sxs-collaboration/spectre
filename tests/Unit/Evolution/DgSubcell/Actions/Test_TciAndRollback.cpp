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
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Actions/TciAndRollback.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/TciGridHistory.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DiscontinuousGalerkin/Tags/NeighborMesh.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Phase.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
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
          evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>,
          evolution::dg::subcell::Tags::TciDecision,
          evolution::dg::subcell::Tags::DataForRdmpTci,
          evolution::dg::Tags::NeighborMesh<Dim>,
          Tags::Variables<tmpl::list<Var1>>,
          Tags::HistoryEvolvedVariables<Tags::Variables<tmpl::list<Var1>>>,
          SelfStart::Tags::InitialValue<Tags::Variables<tmpl::list<Var1>>>,
          evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>>,
      tmpl::conditional_t<
          Metavariables::has_prims,
          tmpl::list<Tags::Variables<tmpl::list<PrimVar1>>,
                     SelfStart::Tags::InitialValue<
                         Tags::Variables<tmpl::list<PrimVar1>>>>,
          tmpl::list<>>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
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
      tmpl::list<evolution::dg::subcell::Tags::SubcellOptions<Dim>>;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool rdmp_fails;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool tci_fails;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool tci_invoked;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool expected_evolve_on_dg_after_tci_failure;

  struct SubcellOptions {
    static constexpr bool subcell_enabled_at_external_boundary = false;

    template <typename DbTagsList>
    static size_t ghost_zone_size(const db::DataBox<DbTagsList>& box) {
      CHECK(db::get<domain::Tags::Mesh<Dim>>(box) ==
            Mesh<Dim>(5, Spectral::Basis::Legendre,
                      Spectral::Quadrature::GaussLobatto));
      return 2;
    }
  };

  struct TciOnDgGrid {
    using return_tags = tmpl::list<>;
    using argument_tags =
        tmpl::list<Tags::Variables<tmpl::list<Var1>>, domain::Tags::Mesh<Dim>,
                   evolution::dg::subcell::Tags::Mesh<Dim>,
                   evolution::dg::subcell::Tags::DataForRdmpTci,
                   evolution::dg::subcell::Tags::SubcellOptions<Dim>>;

    static std::tuple<int, evolution::dg::subcell::RdmpTciData> apply(
        const Variables<tmpl::list<Var1>>& dg_vars, const Mesh<Dim>& dg_mesh,
        const Mesh<Dim>& subcell_mesh,
        const evolution::dg::subcell::RdmpTciData& past_rdmp_data,
        const evolution::dg::subcell::SubcellOptions& subcell_options,
        const double persson_exponent,
        const bool evolve_on_dg_after_tci_failure) {
      // match with global static variable in metavariables

      // assign value of passed in variable
      using metavars = Metavariables<Dim, HasPrims>;

      Variables<tmpl::list<Var1>> projected_vars{
          subcell_mesh.number_of_grid_points()};
      evolution::dg::subcell::fd::project(
          make_not_null(&projected_vars), dg_vars, dg_mesh,
          evolution::dg::subcell::fd::mesh(dg_mesh).extents());
      // Set RDMP TCI data
      using std::max;
      using std::min;
      evolution::dg::subcell::RdmpTciData rdmp_data{};
      rdmp_data.max_variables_values = DataVector{max(
          max(get(get<Var1>(dg_vars))), max(get(get<Var1>(projected_vars))))};
      rdmp_data.min_variables_values = DataVector{min(
          min(get(get<Var1>(dg_vars))), min(get(get<Var1>(projected_vars))))};

      CHECK(approx(persson_exponent) == 4.0);
      CHECK(evolve_on_dg_after_tci_failure ==
            metavars::expected_evolve_on_dg_after_tci_failure);
      tci_invoked = true;
      const bool rdmp_result =
          static_cast<bool>(evolution::dg::subcell::rdmp_tci(
              rdmp_data.max_variables_values, rdmp_data.min_variables_values,
              past_rdmp_data.max_variables_values,
              past_rdmp_data.min_variables_values,
              subcell_options.rdmp_delta0(), subcell_options.rdmp_epsilon()));
      const int decision = rdmp_result ? 10 : (tci_fails ? 5 : 0);
      return {decision, std::move(rdmp_data)};
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
template <size_t Dim, bool HasPrims>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool Metavariables<Dim, HasPrims>::expected_evolve_on_dg_after_tci_failure =
    false;

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

template <size_t Dim>
class TestCreator : public DomainCreator<Dim> {
  Domain<Dim> create_domain() const override { return Domain<Dim>{}; }
  std::vector<DirectionMap<
      Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override {
    return {};
  }

  std::vector<std::string> block_names() const override {
    return {"Block0", "Block1"};
  }

  std::vector<std::array<size_t, Dim>> initial_extents() const override {
    return {};
  }

  std::vector<std::array<size_t, Dim>> initial_refinement_levels()
      const override {
    return {};
  }
};

template <size_t Dim, bool HasPrims>
void test_impl(const bool rdmp_fails, const bool tci_fails,
               const bool always_use_subcell, const bool self_starting,
               const bool with_neighbors, const bool use_halo,
               const bool neighbor_is_troubled,
               const bool disable_subcell_in_block) {
  CAPTURE(Dim);
  CAPTURE(rdmp_fails);
  CAPTURE(tci_fails);
  CAPTURE(always_use_subcell);
  CAPTURE(self_starting);
  CAPTURE(with_neighbors);
  CAPTURE(use_halo);
  CAPTURE(neighbor_is_troubled);
  CAPTURE(disable_subcell_in_block);

  using metavars = Metavariables<Dim, HasPrims>;
  metavars::rdmp_fails = rdmp_fails;
  metavars::tci_fails = tci_fails;
  metavars::tci_invoked = false;

  // Sets neighboring block "Block1" to DG-only, if disable_subcell_in_block ==
  // true.
  using comp = component<Dim, metavars>;
  const evolution::dg::subcell::SubcellOptions& subcell_options =
      evolution::dg::subcell::SubcellOptions{
          evolution::dg::subcell::SubcellOptions{
              1.0e-3, 1.0e-4, 2.0e-3, 2.0e-4, 5.0, 4.0, always_use_subcell,
              evolution::dg::subcell::fd::ReconstructionMethod::DimByDim,
              use_halo,
              disable_subcell_in_block
                  ? std::optional{std::vector<std::string>{"Block1"}}
                  : std::optional<std::vector<std::string>>{},
              ::fd::DerivativeOrder::Two},
          TestCreator<Dim>{}};

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{{subcell_options}};

  const TimeStepId time_step_id{false, self_starting ? -1 : 1,
                                Slab{1.0, 2.0}.end()};
  const TimeDelta step_size{Slab{1.0, 2.0}, {-1, 10}};
  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const Element<Dim> element = create_element<Dim>(with_neighbors);
  const evolution::dg::subcell::ActiveGrid active_grid =
      evolution::dg::subcell::ActiveGrid::Dg;

  using GhostData = evolution::dg::subcell::GhostData;

  FixedHashMap<maximum_number_of_neighbors(Dim),
               std::pair<Direction<Dim>, ElementId<Dim>>, GhostData,
               boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
      ghost_data{};

  FixedHashMap<maximum_number_of_neighbors(Dim),
               std::pair<Direction<Dim>, ElementId<Dim>>, Mesh<Dim>,
               boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
      neighbor_meshes{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    REQUIRE(not neighbors.ids().empty());
    const std::pair directional_element_id{direction, *neighbors.ids().begin()};
    if ((direction.side() == Side::Upper and direction.dimension() % 2 == 0) or
        (direction.side() == Side::Lower and direction.dimension() % 2 != 0)) {
      neighbor_meshes[directional_element_id] = dg_mesh;
      ghost_data[directional_element_id] = GhostData{1};
      DataVector& neighbor_data = ghost_data.at(directional_element_id)
                                      .neighbor_ghost_data_for_reconstruction();
      neighbor_data = DataVector{dg_mesh.number_of_grid_points()};
      alg::iota(neighbor_data, subcell_mesh.number_of_grid_points() *
                                   (2.0 * direction.dimension() + 1.0));
    } else {
      neighbor_meshes[directional_element_id] = subcell_mesh;
      ghost_data[directional_element_id] = GhostData{1};
      DataVector& neighbor_data = ghost_data.at(directional_element_id)
                                      .neighbor_ghost_data_for_reconstruction();
      neighbor_data = DataVector{dg_mesh.number_of_grid_points()};
      alg::iota(neighbor_data, subcell_mesh.number_of_grid_points() *
                                   (2.0 * direction.dimension() + 1.0));
    }
  }
  // test FD/DG element neighbor disable_subcell_in_block
  const bool bordering_dg_block = alg::any_of(
      element.neighbors(),
      [&subcell_options](const auto& direction_and_neighbor) {
        const size_t first_block_id =
            direction_and_neighbor.second.ids().begin()->block_id();
        return alg::found(subcell_options.only_dg_block_ids(), first_block_id);
      });

  const bool self_block_dg_only = std::binary_search(
      subcell_options.only_dg_block_ids().begin(),
      subcell_options.only_dg_block_ids().end(), element.id().block_id());

  // assign value of passed in variable.  Used as a test in apply() above
  metavars::expected_evolve_on_dg_after_tci_failure =
      bordering_dg_block or self_block_dg_only;

  const int tci_decision{-1};

  // max and min of +-2 at last time level means reconstructed vars will be in
  // limit
  evolution::dg::subcell::RdmpTciData rdmp_tci_data{{2.0}, {-2.0}};
  // Make a copy of the RDMP data because in the case where the TCI fails the
  // RDMP TCI data in the DataBox shouldn't have changed.
  const evolution::dg::subcell::RdmpTciData initial_rdmp_tci_data =
      rdmp_tci_data;

  using evolved_vars_tags = tmpl::list<Var1>;
  using dt_evolved_vars_tags = db::wrap_tags_in<Tags::dt, evolved_vars_tags>;
  Variables<evolved_vars_tags> evolved_vars{dg_mesh.number_of_grid_points()};
  // Set Var1 to the logical coords, since those are linear
  get(get<Var1>(evolved_vars)) = get<0>(logical_coordinates(dg_mesh));
  if (rdmp_fails) {
    get(get<Var1>(evolved_vars))[0] = 100.0;
  }
  using prim_vars_tags = tmpl::list<PrimVar1>;
  Variables<prim_vars_tags> prim_vars{dg_mesh.number_of_grid_points()};
  get(get<PrimVar1>(prim_vars)) = get<0>(logical_coordinates(dg_mesh)) + 1000.0;

  constexpr size_t history_size = 5;
  constexpr size_t history_substeps = 3;
  TimeSteppers::History<Variables<evolved_vars_tags>> time_stepper_history{
      history_size};
  {
    Time step_time{Slab{1.0, 2.0}, {6, 10}};
    for (size_t i = 0; i < history_size; ++i) {
      step_time += step_size;
      Variables<dt_evolved_vars_tags> dt_vars{dg_mesh.number_of_grid_points()};
      get(get<Tags::dt<Var1>>(dt_vars)) =
          (i + 20.0) * get<0>(logical_coordinates(dg_mesh));
      time_stepper_history.insert({false, 1, step_time}, i * evolved_vars,
                                  dt_vars);
    }
    for (size_t i = 0; i < history_substeps; ++i) {
      Variables<dt_evolved_vars_tags> dt_vars{dg_mesh.number_of_grid_points()};
      get(get<Tags::dt<Var1>>(dt_vars)) =
          (i + 40.0) * get<0>(logical_coordinates(dg_mesh));
      time_stepper_history.insert(
          {false, 1, step_time, i + 1, step_size, step_time.value()},
          -i * evolved_vars, dt_vars);
    }
    time_stepper_history.discard_value(time_stepper_history[2].time_step_id);
    time_stepper_history.discard_value(
        time_stepper_history.substeps()[1].time_step_id);
  }
  const Variables<evolved_vars_tags> vars = time_stepper_history.latest_value();

  const bool did_rollback = false;
  Variables<evolved_vars_tags> initial_value_evolved_vars{
      dg_mesh.number_of_grid_points(), 1.0e8};
  Variables<prim_vars_tags> initial_value_prim_vars{
      dg_mesh.number_of_grid_points(), 1.0e10};

  typename evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>::type
      neighbor_decisions{};
  neighbor_decisions.insert(
      std::pair{std::pair{Direction<Dim>::lower_xi(), ElementId<Dim>{10}},
                neighbor_is_troubled ? 10 : 0});

  if constexpr (HasPrims) {
    ActionTesting::emplace_array_component_and_initialize<comp>(
        &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0,
        {time_step_id, dg_mesh, subcell_mesh, element, active_grid,
         did_rollback, ghost_data, tci_decision, rdmp_tci_data, neighbor_meshes,
         evolved_vars, time_stepper_history, initial_value_evolved_vars,
         neighbor_decisions, prim_vars, initial_value_prim_vars});
  } else {
    (void)prim_vars;
    (void)initial_value_prim_vars;
    ActionTesting::emplace_array_component_and_initialize<comp>(
        &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0,
        {time_step_id, dg_mesh, subcell_mesh, element, active_grid,
         did_rollback, ghost_data, tci_decision, rdmp_tci_data, neighbor_meshes,
         evolved_vars, time_stepper_history, initial_value_evolved_vars,
         neighbor_decisions});
  }

  // Invoke the TciAndRollback action on the runner
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
      with_neighbors and ((always_use_subcell or rdmp_fails or tci_fails or
                           (use_halo and neighbor_is_troubled)) and
                          not disable_subcell_in_block);

  if (expected_rollback) {
    CHECK(active_grid_from_box == evolution::dg::subcell::ActiveGrid::Subcell);
    CHECK(did_rollback_from_box);
    CHECK(ActionTesting::get_next_action_index<comp>(runner, 0) == 4);

    CHECK(ActionTesting::get_databox_tag<
              comp, evolution::dg::subcell::Tags::DataForRdmpTci>(runner, 0) ==
          initial_rdmp_tci_data);

    CHECK(time_stepper_history_from_box.size() == history_size);
    CHECK(time_stepper_history_from_box.substeps().size() ==
          history_substeps - 1);
    CHECK(time_stepper_history_from_box.integration_order() ==
          time_stepper_history.integration_order());
    {
      const auto check_box_record = [&](const auto& original_record) {
        const auto& record_from_box =
            time_stepper_history_from_box[original_record.time_step_id];
        CHECK(record_from_box.derivative ==
              evolution::dg::subcell::fd::project(
                  original_record.derivative, dg_mesh, subcell_mesh.extents()));
        if (original_record.value.has_value()) {
          CHECK(record_from_box.value ==
                std::optional{evolution::dg::subcell::fd::project(
                    *original_record.value, dg_mesh, subcell_mesh.extents())});
        } else {
          CHECK(not record_from_box.value.has_value());
        }
      };
      for (const auto& original_record : time_stepper_history) {
        check_box_record(original_record);
      }
      for (size_t i = 0; i < history_substeps - 1; ++i) {
        check_box_record(time_stepper_history.substeps()[i]);
      }
    }

    CHECK(active_vars_from_box == evolution::dg::subcell::fd::project(
                                      vars, dg_mesh, subcell_mesh.extents()));

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

    auto expected_ghost_data = ghost_data;
    for (const auto& [directional_element_id, neighbor_mesh] :
         neighbor_meshes) {
      evolution::dg::subcell::insert_or_update_neighbor_volume_data<false>(
          make_not_null(&expected_ghost_data),
          expected_ghost_data.at(directional_element_id)
              .neighbor_ghost_data_for_reconstruction(),
          0, directional_element_id, neighbor_mesh, element, subcell_mesh, 2);
    }
    const auto& ghost_data_for_reconstruction = ActionTesting::get_databox_tag<
        comp, evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>>(
        runner, 0);
    for (const auto& [id, local_ghost_data] : expected_ghost_data) {
      CHECK(ghost_data_for_reconstruction.contains(id));
      CHECK_ITERABLE_APPROX(
          (ghost_data_for_reconstruction.at(id)
               .neighbor_ghost_data_for_reconstruction()),
          local_ghost_data.neighbor_ghost_data_for_reconstruction());
    }

  } else {
    CHECK(ActionTesting::get_next_action_index<comp>(runner, 0) == 2);
    CHECK(active_grid_from_box == evolution::dg::subcell::ActiveGrid::Dg);
    CHECK_FALSE(did_rollback_from_box);

    const auto subcell_vars = evolution::dg::subcell::fd::project(
        evolved_vars, dg_mesh,
        evolution::dg::subcell::fd::mesh(dg_mesh).extents());
    const evolution::dg::subcell::RdmpTciData expected_rdmp_data{
        {std::max(max(get(get<Var1>(evolved_vars))),
                  max(get(get<Var1>(subcell_vars))))},
        {std::min(min(get(get<Var1>(evolved_vars))),
                  min(get(get<Var1>(subcell_vars))))}};

    CHECK(ActionTesting::get_databox_tag<
              comp, evolution::dg::subcell::Tags::DataForRdmpTci>(runner, 0) ==
          expected_rdmp_data);

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

    const auto& ghost_data_for_reconstruction = ActionTesting::get_databox_tag<
        comp, evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>>(
        runner, 0);
    CHECK(ghost_data_for_reconstruction.empty());
  }
  CHECK(ActionTesting::get_databox_tag<
            comp, evolution::dg::subcell::Tags::TciDecision>(runner, 0) ==
        (metavars::tci_invoked ? (rdmp_fails ? 10 : (tci_fails ? 5 : 0)) : -1));
}

template <size_t Dim>
void test() {
  for (const auto& [rdmp_fails, tci_fails, always_use_subcell, self_starting,
                    have_neighbors, use_halo, neighbor_is_troubled,
                    disable_subcell_in_block] :
       cartesian_product(make_array(false, true), make_array(false, true),
                         make_array(false, true), make_array(false, true),
                         make_array(false, true), make_array(false, true),
                         make_array(false, true), make_array(false, true))) {
    test_impl<Dim, true>(rdmp_fails, tci_fails, always_use_subcell,
                         self_starting, have_neighbors, use_halo,
                         neighbor_is_troubled, disable_subcell_in_block);
    test_impl<Dim, false>(rdmp_fails, tci_fails, always_use_subcell,
                          self_starting, have_neighbors, use_halo,
                          neighbor_is_troubled, disable_subcell_in_block);
  }
}

// [[TimeOut, 10]]
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
