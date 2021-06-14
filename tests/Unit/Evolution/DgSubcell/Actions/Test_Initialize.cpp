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
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Actions/Initialize.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/TciGridHistory.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DgSubcell/TciStatus.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct SystemAnalyticSolution : public MarkAsAnalyticSolution {
  template <size_t Dim>
  tuples::TaggedTuple<Var1> variables(
      const tnsr::I<DataVector, Dim>& x, const double t,
      tmpl::list<Var1> /*meta*/) const noexcept {
    tuples::TaggedTuple<Var1> vars(x.get(0) + t);
    for (size_t d = 1; d < Dim; ++d) {
      get(get<Var1>(vars)) += x.get(d) + t;
    }
    return vars;
  }

  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <size_t Dim>
struct System {
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = Dim;
  using variables_tag = Tags::Variables<tmpl::list<Var1>>;
};

template <size_t Dim, typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;

  using initial_tags =
      tmpl::list<Initialization::Tags::InitialTime, domain::Tags::Mesh<Dim>,
                 domain::Tags::Element<Dim>, domain::Tags::FunctionsOfTime,
                 domain::Tags::Coordinates<Dim, Frame::Logical>,
                 domain::Tags::ElementMap<Dim, Frame::Grid>,
                 domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                             Frame::Inertial>,
                 Tags::Variables<tmpl::list<Var1>>,
                 Tags::Variables<tmpl::list<::Tags::dt<Var1>>>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<
          ActionTesting::InitializeDataBox<initial_tags>,
          ::Actions::SetupDataBox,
          evolution::Initialization::Actions::SetVariables<
              domain::Tags::Coordinates<Dim, Frame::Logical>>,
          evolution::dg::subcell::Actions::Initialize<
              Dim, System<Dim>, typename Metavariables::DgInitialDataTci>>>>;
};

template <size_t Dim, bool TciFails>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using analytic_solution = SystemAnalyticSolution;
  using component_list = tmpl::list<Component<Dim, Metavariables>>;
  using system = System<Dim>;
  using analytic_variables_tags = typename system::variables_tag::tags_list;
  using const_global_cache_tags =
      tmpl::list<Tags::AnalyticSolution<analytic_solution>>;
  enum class Phase { Initialization, Exit };
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept {}

  struct DgInitialDataTci {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    static bool invoked;

    using argument_tags = tmpl::list<domain::Tags::Mesh<volume_dim>>;

    static bool apply(
        const Variables<tmpl::list<Var1>>& dg_vars,
        const Variables<
            tmpl::list<evolution::dg::subcell::Tags::Inactive<Var1>>>&
            subcell_vars,
        const double rdmp_delta0, const double rdmp_epsilon,
        const double persson_exponent,
        const Mesh<volume_dim>& dg_mesh) noexcept {
      CHECK(dg_mesh == Mesh<Dim>(5, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto));
      CHECK(rdmp_delta0 == 1.0e-3);
      CHECK(rdmp_epsilon == 1.0e-4);
      CHECK(persson_exponent == 4.0);
      Variables<tmpl::list<evolution::dg::subcell::Tags::Inactive<Var1>>>
          projected_dg_vars{subcell_vars.number_of_grid_points()};
      evolution::dg::subcell::fd::project(
          make_not_null(&projected_dg_vars), dg_vars, dg_mesh,
          evolution::dg::subcell::fd::mesh(dg_mesh).extents());
      CHECK(projected_dg_vars == subcell_vars);
      invoked = true;
      return TciFails;
    }
  };
};

template <size_t Dim, bool TciFails>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool Metavariables<Dim, TciFails>::DgInitialDataTci::invoked = false;

template <size_t Dim, bool TciFails>
void test(const bool always_use_subcell, const bool interior_element) {
  CAPTURE(Dim);
  CAPTURE(TciFails);
  CAPTURE(always_use_subcell);
  CAPTURE(interior_element);
  using metavars = Metavariables<Dim, TciFails>;
  using comp = Component<Dim, metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{
      {SystemAnalyticSolution{},
       evolution::dg::subcell::SubcellOptions{1.0e-3, 1.0e-4, 2.0e-3, 2.0e-4,
                                              4.0, 4.1, always_use_subcell}}};
  Metavariables<Dim, TciFails>::DgInitialDataTci::invoked = false;

  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const ElementId<Dim> self_id{0};
  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  if (always_use_subcell and interior_element) {
    size_t id_count = 1;
    for (const auto& direction : Direction<Dim>::all_directions()) {
      neighbors[direction] = Neighbors<Dim>{{ElementId<Dim>{id_count}}, {}};
      ++id_count;
    }
  }
  const Element<Dim> element{self_id, neighbors};
  const auto logical_coords = logical_coordinates(dg_mesh);
  ElementMap<Dim, Frame::Grid> logical_to_grid_map{
      self_id, domain::make_coordinate_map_base<Frame::Logical, Frame::Grid>(
                   domain::CoordinateMaps::Identity<Dim>{})};
  const auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<Dim>{});
  const double initial_time = 1.3;
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  const Variables<tmpl::list<Var1>> var(get<0>(logical_coords).size(), 8.9999);

  ActionTesting::emplace_array_component_and_initialize<comp>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0,
      {initial_time, dg_mesh, element, clone_unique_ptrs(functions_of_time),
       logical_coords, std::move(logical_to_grid_map),
       grid_to_inertial_map->get_clone(), var,
       Variables<tmpl::list<::Tags::dt<Var1>>>{
           dg_mesh.number_of_grid_points()}});
  // Invoke the SetupDataBox action on the runner
  ActionTesting::next_action<comp>(make_not_null(&runner), 0);

  // Invoke the SetVariables action on the runner
  ActionTesting::next_action<comp>(make_not_null(&runner), 0);

  // Invoke the Initialize action on the runner
  ActionTesting::next_action<comp>(make_not_null(&runner), 0);

  REQUIRE(Metavariables<Dim, TciFails>::DgInitialDataTci::invoked);
  CHECK(
      ActionTesting::get_databox_tag<comp,
                                     evolution::dg::subcell::Tags::DidRollback>(
          runner, 0) == false);
  CHECK(
      ActionTesting::get_databox_tag<comp,
                                     evolution::dg::subcell::Tags::ActiveGrid>(
          runner, 0) == (TciFails or (always_use_subcell and interior_element)
                             ? evolution::dg::subcell::ActiveGrid::Subcell
                             : evolution::dg::subcell::ActiveGrid::Dg));
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  CHECK(ActionTesting::get_databox_tag<comp,
                                       evolution::dg::subcell::Tags::Mesh<Dim>>(
            runner, 0) == subcell_mesh);

  if (TciFails or (always_use_subcell and interior_element)) {
    Variables<tmpl::list<Var1>> subcell_vars{
        subcell_mesh.number_of_grid_points()};
    evolution::dg::subcell::fd::project(
        make_not_null(&subcell_vars),
        ActionTesting::get_databox_tag<
            comp, evolution::dg::subcell::Tags::Inactive<
                      typename System<Dim>::variables_tag>>(runner, 0),
        dg_mesh, subcell_mesh.extents());
    CHECK_ITERABLE_APPROX(
        get<Var1>(ActionTesting::get_databox_tag<
                  comp, typename System<Dim>::variables_tag>(runner, 0)),
        get<Var1>(subcell_vars));
  } else {
    Variables<tmpl::list<evolution::dg::subcell::Tags::Inactive<Var1>>>
        subcell_vars{subcell_mesh.number_of_grid_points()};
    evolution::dg::subcell::fd::project(
        make_not_null(&subcell_vars),
        ActionTesting::get_databox_tag<comp,
                                       typename System<Dim>::variables_tag>(
            runner, 0),
        dg_mesh, subcell_mesh.extents());
    CHECK_ITERABLE_APPROX(
        get<evolution::dg::subcell::Tags::Inactive<Var1>>(
            ActionTesting::get_databox_tag<
                comp, evolution::dg::subcell::Tags::Inactive<
                          typename System<Dim>::variables_tag>>(runner, 0)),
        get<evolution::dg::subcell::Tags::Inactive<Var1>>(subcell_vars));
  }
  CHECK(ActionTesting::tag_is_retrievable<
        comp, evolution::dg::subcell::Tags::DidRollback>(runner, 0));
  CHECK(ActionTesting::tag_is_retrievable<
        comp, evolution::dg::subcell::Tags::TciGridHistory>(runner, 0));
  CHECK(ActionTesting::tag_is_retrievable<
        comp, evolution::dg::subcell::Tags::
                  NeighborDataForReconstructionAndRdmpTci<Dim>>(runner, 0));
  CHECK(ActionTesting::tag_is_retrievable<
        comp,
        evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToGrid<Dim>>(
      runner, 0));
  CHECK(ActionTesting::tag_is_retrievable<
        comp,
        evolution::dg::subcell::fd::Tags::DetInverseJacobianLogicalToGrid>(
      runner, 0));
  CHECK(ActionTesting::tag_is_retrievable<
        comp, evolution::dg::subcell::Tags::Coordinates<Dim, Frame::Logical>>(
      runner, 0));
  CHECK(ActionTesting::tag_is_retrievable<
        comp, evolution::dg::subcell::Tags::Coordinates<Dim, Frame::Grid>>(
      runner, 0));
  CHECK(ActionTesting::tag_is_retrievable<
        comp, evolution::dg::subcell::Tags::TciStatus>(runner, 0));
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Actions.Initialize",
                  "[Evolution][Unit]") {
  Parallel::register_classes_with_charm<
      domain::CoordinateMap<Frame::Logical, Frame::Grid,
                            domain::CoordinateMaps::Identity<1>>,
      domain::CoordinateMap<Frame::Logical, Frame::Grid,
                            domain::CoordinateMaps::Identity<2>>,
      domain::CoordinateMap<Frame::Logical, Frame::Grid,
                            domain::CoordinateMaps::Identity<3>>,
      domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                            domain::CoordinateMaps::Identity<1>>,
      domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                            domain::CoordinateMaps::Identity<2>>,
      domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                            domain::CoordinateMaps::Identity<3>>>();
  for (const bool always_use_subcell : {false, true}) {
    for (const bool interior_element : {false, true}) {
      test<1, true>(always_use_subcell, interior_element);
      test<1, false>(always_use_subcell, interior_element);
    }
  }
}
}  // namespace
