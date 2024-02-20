// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Actions/Initialize.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/ReconstructionOrder.hpp"
#include "Evolution/DgSubcell/Tags/StepsSinceTciCall.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/TciCallsSinceRollback.hpp"
#include "Evolution/DgSubcell/Tags/TciGridHistory.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/MockRuntimeSystemFreeFunctions.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct SystemAnalyticSolution : public MarkAsAnalyticSolution {
  template <size_t Dim>
  tuples::TaggedTuple<Var1> variables(const tnsr::I<DataVector, Dim>& x,
                                      const double t,
                                      tmpl::list<Var1> /*meta*/) const {
    tuples::TaggedTuple<Var1> vars(x.get(0) + t);
    for (size_t d = 1; d < Dim; ++d) {
      get(get<Var1>(vars)) += x.get(d) + t;
    }
    return vars;
  }

  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) {}  // NOLINT
};

template <size_t Dim>
struct System {
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = Dim;
  using variables_tag = Tags::Variables<tmpl::list<Var1>>;
  using flux_variables = tmpl::list<Var1>;
};

template <size_t Dim, typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using const_global_cache_tag_list = tmpl::list<>;

  using initial_tags = tmpl::list<
      Tags::Time, domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
      domain::Tags::FunctionsOfTimeInitialize,
      domain::Tags::Coordinates<Dim, Frame::ElementLogical>,
      domain::Tags::ElementMap<Dim, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                  Frame::Inertial>,
      Tags::Variables<tmpl::list<Var1>>,
      Tags::Variables<tmpl::list<::Tags::dt<Var1>>>,
      ::Tags::HistoryEvolvedVariables<Tags::Variables<tmpl::list<Var1>>>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<
          ActionTesting::InitializeDataBox<
              initial_tags,
              db::AddComputeTags<
                  domain::Tags::MappedCoordinates<
                      domain::Tags::ElementMap<Dim, Frame::Grid>,
                      domain::Tags::Coordinates<Dim, Frame::ElementLogical>>,
                  domain::Tags::CoordinatesMeshVelocityAndJacobiansCompute<
                      domain::CoordinateMaps::Tags::CoordinateMap<
                          Dim, Frame::Grid, Frame::Inertial>>,
                  domain::Tags::InertialFromGridCoordinatesCompute<Dim>>>,
          evolution::dg::subcell::Actions::SetSubcellGrid<Dim, System<Dim>,
                                                          false>,

          evolution::dg::subcell::Actions::SetAndCommunicateInitialRdmpData<
              Dim, typename Metavariables::SetInitialRdmpData>,
          evolution::dg::subcell::Actions::ComputeAndSendTciOnInitialGrid<
              Dim, System<Dim>, typename Metavariables::FdInitialDataTci>,
          evolution::dg::subcell::Actions::SetInitialGridFromTciData<
              Dim, System<Dim>>>>>;
};

template <size_t Dim, bool TciFails, bool SubcellEnabledOnExternalBoundary>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using analytic_solution = SystemAnalyticSolution;
  using component_list = tmpl::list<Component<Dim, Metavariables>>;
  using system = System<Dim>;
  using analytic_variables_tags = typename system::variables_tag::tags_list;
  using const_global_cache_tags =
      tmpl::list<Tags::AnalyticSolution<analytic_solution>>;
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}

  struct SubcellOptions {
    static constexpr bool subcell_enabled_at_external_boundary =
        SubcellEnabledOnExternalBoundary;
  };

  struct FdInitialDataTci {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    static bool invoked;

    using return_tags = tmpl::list<>;
    using argument_tags = tmpl::list<::Tags::Variables<tmpl::list<Var1>>,
                                     domain::Tags::Mesh<volume_dim>>;

    static std::tuple<int, evolution::dg::subcell::RdmpTciData> apply(
        const Variables<tmpl::list<Var1>>& fd_vars,
        const Mesh<volume_dim>& dg_mesh, const double persson_exponent,
        const bool need_rdmp_data_only) {
      CHECK(dg_mesh == Mesh<Dim>(5, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto));
      CHECK(persson_exponent == 5.1);
      evolution::dg::subcell::RdmpTciData rdmp_data{};
      rdmp_data.max_variables_values = DataVector{max(get(get<Var1>(fd_vars)))};
      rdmp_data.min_variables_values = DataVector{min(get(get<Var1>(fd_vars)))};
      REQUIRE(not need_rdmp_data_only);
      invoked = true;
      return {static_cast<int>(TciFails), std::move(rdmp_data)};
    }
  };

  struct SetInitialRdmpData {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    static bool invoked;

    using return_tags =
        tmpl::list<evolution::dg::subcell::Tags::DataForRdmpTci>;
    using argument_tags = tmpl::list<::Tags::Variables<tmpl::list<Var1>>>;

    static void apply(
        const gsl::not_null<evolution::dg::subcell::RdmpTciData*> rdmp_tci_data,
        const Variables<tmpl::list<Var1>>& fd_vars) {
      invoked = true;
      evolution::dg::subcell::RdmpTciData rdmp_data{};
      rdmp_data.max_variables_values = DataVector{max(get(get<Var1>(fd_vars)))};
      rdmp_data.min_variables_values = DataVector{min(get(get<Var1>(fd_vars)))};
      *rdmp_tci_data = std::move(rdmp_data);
    }
  };
};

template <size_t Dim, bool TciFails, bool SubcellEnabledOnExternalBoundary>
bool Metavariables<
    Dim, TciFails,
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    SubcellEnabledOnExternalBoundary>::FdInitialDataTci::invoked = false;
template <size_t Dim, bool TciFails, bool SubcellEnabledOnExternalBoundary>
bool Metavariables<Dim, TciFails, SubcellEnabledOnExternalBoundary>::
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    SetInitialRdmpData::invoked = false;

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

template <size_t Dim, bool TciFails, bool SubcellEnabledOnExternalBoundary>
void test(const bool always_use_subcell, const bool interior_element,
          const bool allow_subcell_in_block) {
  CAPTURE(Dim);
  CAPTURE(TciFails);
  CAPTURE(SubcellEnabledOnExternalBoundary);
  CAPTURE(always_use_subcell);
  CAPTURE(interior_element);
  CAPTURE(allow_subcell_in_block);
  using metavars =
      Metavariables<Dim, TciFails, SubcellEnabledOnExternalBoundary>;
  using comp = Component<Dim, metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{
      {SystemAnalyticSolution{},
       evolution::dg::subcell::SubcellOptions{
           evolution::dg::subcell::SubcellOptions{
               4.1, 1_st, 1.0e-3, 1.0e-4, always_use_subcell,
               evolution::dg::subcell::fd::ReconstructionMethod::DimByDim,
               false,
               allow_subcell_in_block
                   ? std::optional<std::vector<std::string>>{}
                   : std::optional{std::vector<std::string>{"Block0"}},
               ::fd::DerivativeOrder::Two},
           TestCreator<Dim>{}}}};
  metavars::FdInitialDataTci::invoked = false;
  metavars::SetInitialRdmpData::invoked = false;

  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const ElementId<Dim> self_id{0};
  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  if (interior_element) {
    size_t id_count = 1;
    for (const auto& direction : Direction<Dim>::all_directions()) {
      neighbors[direction] = Neighbors<Dim>{{ElementId<Dim>{id_count}}, {}};
      ++id_count;
    }
  }
  const Element<Dim> element{self_id, neighbors};
  const auto logical_coords = logical_coordinates(dg_mesh);
  const auto make_element_map = [](const auto& element_id) {
    return ElementMap<Dim, Frame::Grid>{
        element_id,
        domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
            domain::CoordinateMaps::Identity<Dim>{})};
  };
  const auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<Dim>{});
  const double initial_time = 1.3;
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  ActionTesting::emplace_array_component_and_initialize<comp>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, self_id,
      {initial_time, dg_mesh, element, clone_unique_ptrs(functions_of_time),
       logical_coords, make_element_map(self_id),
       grid_to_inertial_map->get_clone(),
       Variables<tmpl::list<Var1>>{subcell_mesh.number_of_grid_points()},
       Variables<tmpl::list<::Tags::dt<Var1>>>{
           subcell_mesh.number_of_grid_points()},
       typename ::Tags::HistoryEvolvedVariables<
           Tags::Variables<tmpl::list<Var1>>>::type{}});

  if (interior_element) {
    // Add neighboring elements into runner
    for (const auto& [direction, neighbors_in_direction] :
         element.neighbors()) {
      DirectionMap<Dim, Neighbors<Dim>> neighbor_neighbors{};
      neighbor_neighbors[direction.opposite()] = Neighbors<Dim>{{self_id}, {}};
      const auto neighbor_id = *neighbors_in_direction.begin();
      // We use the time to get different solutions on the different neighbors
      // since the analytic solution includes the time.
      const double neighbor_time = initial_time + neighbor_id.block_id();

      ActionTesting::emplace_array_component_and_initialize<comp>(
          &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
          neighbor_id,
          {neighbor_time, dg_mesh,
           Element<Dim>{neighbor_id, neighbor_neighbors},
           clone_unique_ptrs(functions_of_time), logical_coords,
           make_element_map(neighbor_id), grid_to_inertial_map->get_clone(),
           Variables<tmpl::list<Var1>>{subcell_mesh.number_of_grid_points()},
           Variables<tmpl::list<::Tags::dt<Var1>>>{
               subcell_mesh.number_of_grid_points()},
           typename ::Tags::HistoryEvolvedVariables<
               Tags::Variables<tmpl::list<Var1>>>::type{}});
    }
  }

  REQUIRE(
      ActionTesting::get_databox_tag<comp, typename System<Dim>::variables_tag>(
          runner, self_id)
          .number_of_grid_points() == subcell_mesh.number_of_grid_points());

  // Invoke SetSubcellGrid action on the runner
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);
  for (size_t count = 1; interior_element and count <= 2 * Dim; ++count) {
    ActionTesting::next_action<comp>(make_not_null(&runner),
                                     ElementId<Dim>{count});
  }
  const auto active_grid_before_tci =
      ActionTesting::get_databox_tag<comp,
                                     evolution::dg::subcell::Tags::ActiveGrid>(
          runner, self_id);
  CHECK(active_grid_before_tci ==
        (((element.external_boundaries().empty() or
           SubcellEnabledOnExternalBoundary) and
          allow_subcell_in_block)
             ? evolution::dg::subcell::ActiveGrid::Subcell
             : evolution::dg::subcell::ActiveGrid::Dg));

  // Invoke SetAndCommunicateInitialRdmpData action on the runner
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);
  REQUIRE(metavars::SetInitialRdmpData::invoked);
  REQUIRE_FALSE(metavars::FdInitialDataTci::invoked);
  if (interior_element) {
    for (size_t count = 1; count <= 2 * Dim; ++count) {
      ActionTesting::next_action<comp>(make_not_null(&runner),
                                       ElementId<Dim>{count});
      if (count < 2 * Dim) {
        REQUIRE(ActionTesting::get_next_action_index<comp>(runner, self_id) ==
                3);
        REQUIRE_FALSE(ActionTesting::next_action_if_ready<comp>(
            make_not_null(&runner), self_id));
        REQUIRE(ActionTesting::get_next_action_index<comp>(runner, self_id) ==
                3);
      }
    }

    // Check RDMP data was communicated correctly by looking at inboxes.
    const auto& self_inbox = ActionTesting::get_inbox_tag<
        comp, evolution::dg::subcell::Tags::InitialTciData<Dim>>(
        make_not_null(&runner), self_id);
    REQUIRE(self_inbox.size() == 1);
    REQUIRE(self_inbox.find(0) != self_inbox.end());
    const auto& self_inbox_at_time = self_inbox.at(0);
    for (const auto& [direction, neighbors_in_direction] :
         element.neighbors()) {
      const DirectionalId<Dim> id{direction, *neighbors_in_direction.begin()};
      CAPTURE(id);
      REQUIRE(self_inbox_at_time.find(id) != self_inbox_at_time.end());
      CHECK_FALSE(self_inbox_at_time.at(id).tci_status.has_value());
      REQUIRE(self_inbox_at_time.at(id).initial_rdmp_data.has_value());
      CHECK(
          self_inbox_at_time.at(id)
              .initial_rdmp_data.value()
              .max_variables_values[0] ==
          approx(((SubcellEnabledOnExternalBoundary and allow_subcell_in_block)
                      ? 2.18888888888888888
                      : 2.3) +
                 neighbors_in_direction.begin()->block_id()));
      CHECK(
          self_inbox_at_time.at(id)
              .initial_rdmp_data.value()
              .min_variables_values[0] ==
          approx(((SubcellEnabledOnExternalBoundary and allow_subcell_in_block)
                      ? 0.41111111111111111
                      : 0.3) +
                 neighbors_in_direction.begin()->block_id()));
    }
  }
  metavars::FdInitialDataTci::invoked = false;
  metavars::SetInitialRdmpData::invoked = false;

  REQUIRE(ActionTesting::get_next_action_index<comp>(runner, self_id) == 3);
  // Invoke ComputeAndSendTciOnInitialGrid action on self_id
  REQUIRE(ActionTesting::next_action_if_ready<comp>(make_not_null(&runner),
                                                    self_id));
  REQUIRE(ActionTesting::get_next_action_index<comp>(runner, self_id) == 4);
  REQUIRE_FALSE(metavars::SetInitialRdmpData::invoked);
  if (always_use_subcell or
      ActionTesting::get_databox_tag<comp,
                                     evolution::dg::subcell::Tags::ActiveGrid>(
          runner, self_id) == evolution::dg::subcell::ActiveGrid::Dg) {
    REQUIRE_FALSE(metavars::FdInitialDataTci::invoked);
  } else {
    REQUIRE(metavars::FdInitialDataTci::invoked);
  }
  if (interior_element) {
    // Invoke ComputeAndSendTciOnInitialGrid action on the neighbors
    for (size_t i = 0; i < neighbors.size(); ++i) {
      CAPTURE(i + 1);
      metavars::FdInitialDataTci::invoked = false;
      metavars::SetInitialRdmpData::invoked = false;
      ActionTesting::next_action<comp>(make_not_null(&runner),
                                       ElementId<Dim>{i + 1});
      REQUIRE_FALSE(metavars::SetInitialRdmpData::invoked);
      if (not SubcellEnabledOnExternalBoundary or always_use_subcell or
          ActionTesting::get_databox_tag<
              comp, evolution::dg::subcell::Tags::ActiveGrid>(
              runner, self_id) == evolution::dg::subcell::ActiveGrid::Dg) {
        REQUIRE_FALSE(metavars::FdInitialDataTci::invoked);
      } else {
        REQUIRE(metavars::FdInitialDataTci::invoked);
      }
    }

    // Check TCI status data was communicated correctly by looking at inboxes.
    const auto& self_inbox = ActionTesting::get_inbox_tag<
        comp, evolution::dg::subcell::Tags::InitialTciData<Dim>>(
        make_not_null(&runner), self_id);
    REQUIRE(self_inbox.size() == 1);
    REQUIRE(self_inbox.find(1) != self_inbox.end());
    const auto& self_inbox_at_time = self_inbox.at(1);
    for (const auto& [direction, neighbors_in_direction] :
         element.neighbors()) {
      const DirectionalId<Dim> id{direction, *neighbors_in_direction.begin()};
      CAPTURE(id);
      REQUIRE(self_inbox_at_time.find(id) != self_inbox_at_time.end());
      CHECK_FALSE(self_inbox_at_time.at(id).initial_rdmp_data.has_value());
      REQUIRE(self_inbox_at_time.at(id).tci_status.has_value());
      if (metavars::FdInitialDataTci::invoked) {
        CHECK(self_inbox_at_time.at(id).tci_status.value() ==
              static_cast<int>(TciFails));
      } else {
        CHECK(self_inbox_at_time.at(id).tci_status.value() == 0);
      }
    }
  }

  metavars::FdInitialDataTci::invoked = false;
  metavars::SetInitialRdmpData::invoked = false;

  REQUIRE(ActionTesting::get_next_action_index<comp>(runner, self_id) == 4);
  // Invoke SetInitialGridFromTciData action on self_id
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);
  REQUIRE_FALSE(metavars::FdInitialDataTci::invoked);
  REQUIRE_FALSE(metavars::SetInitialRdmpData::invoked);

  const auto active_grid =
      ActionTesting::get_databox_tag<comp,
                                     evolution::dg::subcell::Tags::ActiveGrid>(
          runner, self_id);

  if (SubcellEnabledOnExternalBoundary) {
    CHECK(active_grid ==
          (((TciFails or always_use_subcell) and allow_subcell_in_block)
               ? evolution::dg::subcell::ActiveGrid::Subcell
               : evolution::dg::subcell::ActiveGrid::Dg));
  } else {
    CHECK(active_grid == (((TciFails or always_use_subcell) and
                           allow_subcell_in_block and interior_element)
                              ? evolution::dg::subcell::ActiveGrid::Subcell
                              : evolution::dg::subcell::ActiveGrid::Dg));
  }

  CHECK(
      ActionTesting::get_databox_tag<comp, typename System<Dim>::variables_tag>(
          runner, self_id)
          .number_of_grid_points() ==
      (active_grid == evolution::dg::subcell::ActiveGrid::Dg
           ? dg_mesh.number_of_grid_points()
           : subcell_mesh.number_of_grid_points()));

  // Check that tags were added.
  CHECK(ActionTesting::tag_is_retrievable<
        comp, evolution::dg::subcell::Tags::DidRollback>(runner, self_id));
  CHECK(ActionTesting::tag_is_retrievable<
        comp, evolution::dg::subcell::Tags::TciCallsSinceRollback>(runner,
                                                                   self_id));
  CHECK(ActionTesting::tag_is_retrievable<
        comp, evolution::dg::subcell::Tags::StepsSinceTciCall>(runner,
                                                               self_id));
  CHECK(ActionTesting::tag_is_retrievable<
        comp, evolution::dg::subcell::Tags::TciGridHistory>(runner, self_id));
  CHECK(ActionTesting::tag_is_retrievable<
        comp, evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>>(
      runner, self_id));
  CHECK(ActionTesting::tag_is_retrievable<
        comp, evolution::dg::subcell::Tags::DataForRdmpTci>(runner, self_id));
  CHECK(ActionTesting::tag_is_retrievable<
        comp,
        evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToGrid<Dim>>(
      runner, self_id));
  CHECK(ActionTesting::tag_is_retrievable<
        comp,
        evolution::dg::subcell::fd::Tags::DetInverseJacobianLogicalToGrid>(
      runner, self_id));
  CHECK(ActionTesting::tag_is_retrievable<
        comp, evolution::dg::subcell::fd::Tags::
                  InverseJacobianLogicalToInertial<Dim>>(runner, self_id));
  CHECK(ActionTesting::tag_is_retrievable<
        comp,
        evolution::dg::subcell::fd::Tags::DetInverseJacobianLogicalToInertial>(
      runner, self_id));
  CHECK(ActionTesting::tag_is_retrievable<
        comp,
        evolution::dg::subcell::Tags::Coordinates<Dim, Frame::ElementLogical>>(
      runner, self_id));
  CHECK(ActionTesting::tag_is_retrievable<
        comp, evolution::dg::subcell::Tags::Coordinates<Dim, Frame::Grid>>(
      runner, self_id));
  CHECK(ActionTesting::tag_is_retrievable<
        comp, evolution::dg::subcell::Tags::Coordinates<Dim, Frame::Inertial>>(
      runner, self_id));
  CHECK(ActionTesting::tag_is_retrievable<
        comp, evolution::dg::subcell::Tags::TciDecision>(runner, self_id));
  CHECK(ActionTesting::tag_is_retrievable<
        comp, evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>>(
      runner, self_id));
  CHECK(ActionTesting::tag_is_retrievable<
        comp, evolution::dg::subcell::Tags::ReconstructionOrder<Dim>>(runner,
                                                                      self_id));
  CHECK(ActionTesting::tag_is_retrievable<
        comp,
        evolution::dg::subcell::Tags::InterpolatorsFromFdToNeighborFd<Dim>>(
      runner, self_id));
  CHECK(ActionTesting::tag_is_retrievable<
        comp,
        evolution::dg::subcell::Tags::InterpolatorsFromDgToNeighborFd<Dim>>(
      runner, self_id));
  CHECK(ActionTesting::tag_is_retrievable<
        comp,
        evolution::dg::subcell::Tags::InterpolatorsFromNeighborDgToFd<Dim>>(
      runner, self_id));

  // Check things have correct values.
  CHECK(ActionTesting::get_databox_tag<
            comp, evolution::dg::subcell::Tags::TciCallsSinceRollback>(
            runner, self_id) == 0);
  CHECK(ActionTesting::get_databox_tag<
            comp, evolution::dg::subcell::Tags::StepsSinceTciCall>(
            runner, self_id) == 0);
  CHECK(ActionTesting::get_databox_tag<
            comp, evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>>(
            runner, self_id)
            .size() ==
        (interior_element ? Direction<Dim>::all_directions().size() : 0));
  CHECK(
      ActionTesting::get_databox_tag<comp,
                                     evolution::dg::subcell::Tags::DidRollback>(
          runner, self_id) == false);
  CHECK(ActionTesting::get_databox_tag<comp,
                                       evolution::dg::subcell::Tags::Mesh<Dim>>(
            runner, self_id) == subcell_mesh);
  for (const auto& [direction, neighbors_in_direction] : element.neighbors()) {
    for (const auto& neighbor : neighbors_in_direction.ids()) {
      CHECK(ActionTesting::get_databox_tag<
                comp, evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>>(
                runner, self_id)
                .contains(DirectionalId<Dim>{direction, neighbor}));
    }
  }
  const auto& subcell_inertial_coords = ActionTesting::get_databox_tag<
      comp, evolution::dg::subcell::Tags::Coordinates<Dim, Frame::Inertial>>(
      runner, self_id);
  const auto subcell_logical_coords = logical_coordinates(subcell_mesh);
  for (size_t d = 0; d < Dim; ++d) {
    CHECK(subcell_inertial_coords[d] == subcell_logical_coords[d]);
  }
  CHECK(not ActionTesting::get_databox_tag<
                comp, evolution::dg::subcell::Tags::CellCenteredFlux<
                          typename metavars::system::flux_variables, Dim>>(
                runner, self_id)
                .has_value());

  // Update the variables with the latest in the DataBox
  const auto& vars =
      ActionTesting::get_databox_tag<comp, typename System<Dim>::variables_tag>(
          runner, self_id);
  Variables<tmpl::list<Var1>> expected_vars{};
  if (active_grid == evolution::dg::subcell::ActiveGrid::Subcell) {
    expected_vars.initialize(subcell_mesh.number_of_grid_points());
  } else {
    expected_vars.initialize(dg_mesh.number_of_grid_points());
  }
  expected_vars.assign_subset(SystemAnalyticSolution{}.variables(
      active_grid == evolution::dg::subcell::ActiveGrid::Subcell
          ? ActionTesting::get_databox_tag<
                comp, evolution::dg::subcell::Tags::Coordinates<
                          Dim, Frame::Inertial>>(runner, self_id)
          : ActionTesting::get_databox_tag<
                comp, domain::Tags::Coordinates<Dim, Frame::Inertial>>(runner,
                                                                       self_id),
      ActionTesting::get_databox_tag<comp, ::Tags::Time>(runner, self_id),
      tmpl::list<Var1>{}));
  CHECK_ITERABLE_APPROX(get<Var1>(vars), get<Var1>(expected_vars));
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Actions.Initialize",
                  "[Evolution][Unit]") {
  register_classes_with_charm<
      domain::CoordinateMap<Frame::BlockLogical, Frame::Grid,
                            domain::CoordinateMaps::Identity<1>>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Grid,
                            domain::CoordinateMaps::Identity<2>>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Grid,
                            domain::CoordinateMaps::Identity<3>>,
      domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                            domain::CoordinateMaps::Identity<1>>,
      domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                            domain::CoordinateMaps::Identity<2>>,
      domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                            domain::CoordinateMaps::Identity<3>>>();
  for (const bool always_use_subcell : {false, true}) {
    for (const bool interior_element : {false, true}) {
      for (const bool allow_subcell_in_block : {false, true}) {
        test<1, true, false>(always_use_subcell, interior_element,
                             allow_subcell_in_block);
        test<1, false, false>(always_use_subcell, interior_element,
                              allow_subcell_in_block);

        test<1, true, true>(always_use_subcell, interior_element,
                            allow_subcell_in_block);
        test<1, false, true>(always_use_subcell, interior_element,
                             allow_subcell_in_block);
      }
    }
  }
}
}  // namespace
