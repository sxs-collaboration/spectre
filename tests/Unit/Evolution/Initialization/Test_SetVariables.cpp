// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct TimeId : db::SimpleTag {
  using type = double;
};

struct Var : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct PrimVar : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct EquationOfStateTag : db::SimpleTag {
  using type = double;
};

struct SystemAnalyticSolution : public MarkAsAnalyticSolution {
  template <size_t Dim>
  tuples::TaggedTuple<Var> variables(const tnsr::I<DataVector, Dim>& x,
                                     const double t,
                                     tmpl::list<Var> /*meta*/) const noexcept {
    tuples::TaggedTuple<Var> vars(x.get(0) + t);
    for (size_t d = 1; d < Dim; ++d) {
      get(get<Var>(vars)) += x.get(d) + t;
    }
    return vars;
  }

  template <size_t Dim>
  tuples::TaggedTuple<PrimVar> variables(const tnsr::I<DataVector, Dim>& x,
                                         const double t,
                                         tmpl::list<PrimVar> /*meta*/) const
      noexcept {
    tuples::TaggedTuple<PrimVar> vars(2.0 * x.get(0) + t);
    for (size_t d = 1; d < Dim; ++d) {
      get(get<PrimVar>(vars)) += 2.0 * x.get(d) + t;
    }
    return vars;
  }

  // EoS just needs to be a dummy place holder
  static double equation_of_state() noexcept { return 7.0; }

  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

struct SystemAnalyticData : public MarkAsAnalyticData {
  template <size_t Dim>
  tuples::TaggedTuple<Var> variables(const tnsr::I<DataVector, Dim>& x,
                                     tmpl::list<Var> /*meta*/) const noexcept {
    tuples::TaggedTuple<Var> vars(x.get(0));
    for (size_t d = 1; d < Dim; ++d) {
      get(get<Var>(vars)) += square(x.get(d));
    }
    return vars;
  }

  template <size_t Dim>
  tuples::TaggedTuple<PrimVar> variables(const tnsr::I<DataVector, Dim>& x,
                                         tmpl::list<PrimVar> /*meta*/) const
      noexcept {
    tuples::TaggedTuple<PrimVar> vars(2.0 * x.get(0));
    for (size_t d = 1; d < Dim; ++d) {
      get(get<PrimVar>(vars)) += square(2.0 * x.get(d));
    }
    return vars;
  }

  // EoS just needs to be a dummy place holder
  static double equation_of_state() noexcept { return 7.0; }

  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <size_t Dim, bool HasPrimitiveAndConservativeVars>
struct System {
  // is_in_flux_conservative_form is unused
  static constexpr bool is_in_flux_conservative_form = false;
  static constexpr bool has_primitive_and_conservative_vars =
      HasPrimitiveAndConservativeVars;
  static constexpr size_t volume_dim = Dim;
  using variables_tag = Tags::Variables<tmpl::list<Var>>;
  using primitive_variables_tag = Tags::Variables<tmpl::list<PrimVar>>;
};

template <size_t Dim, typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;

  using initial_tags = tmpl::list<
      Initialization::Tags::InitialTime, domain::Tags::FunctionsOfTime,
      domain::Tags::Coordinates<Dim, Frame::Logical>,
      domain::Tags::ElementMap<Dim, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                  Frame::Inertial>,
      Tags::Variables<tmpl::list<Var>>, Tags::Variables<tmpl::list<PrimVar>>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<initial_tags>,
                 evolution::Initialization::Actions::SetVariables<
                     domain::Tags::Coordinates<Dim, Frame::Logical>>>>>;
};

template <size_t Dim, typename Metavariables>
auto emplace_component(
    const gsl::not_null<ActionTesting::MockRuntimeSystem<Metavariables>*>
        runner,
    const double initial_time) {
  using comp = component<Dim, Metavariables>;

  const auto logical_coords = logical_coordinates(Mesh<Dim>{
      5, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto});
  ElementMap<Dim, Frame::Grid> logical_to_grid_map{
      ElementId<Dim>{0},
      domain::make_coordinate_map_base<Frame::Logical, Frame::Grid>(
          domain::CoordinateMaps::Identity<Dim>{})};
  const std::string expansion_factor = "Expansion";
  const auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::TimeDependent::CubicScale<Dim>{
              10.0, expansion_factor, expansion_factor});

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time.insert(std::make_pair(
      expansion_factor,
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time, std::array<DataVector, 3>{{{1.0}, {-0.1}, {0.0}}})));
  Variables<tmpl::list<Var>> var(get<0>(logical_coords).size(), 8.9999);
  Variables<tmpl::list<PrimVar>> prim_var(get<0>(logical_coords).size(),
                                          9.9999);
  ActionTesting::emplace_component_and_initialize<comp>(
      runner, 0,
      {initial_time, clone_unique_ptrs(functions_of_time), logical_coords,
       std::move(logical_to_grid_map), grid_to_inertial_map->get_clone(), var,
       prim_var});
  return (*grid_to_inertial_map)(
      ActionTesting::get_databox_tag<
          comp, domain::Tags::ElementMap<Dim, Frame::Grid>>(*runner,
                                                            0)(logical_coords),
      initial_time, functions_of_time);
}

template <size_t Dim, bool HasPrimitives>
struct MetavariablesAnalyticSolution {
  static constexpr size_t volume_dim = Dim;
  using analytic_solution = SystemAnalyticSolution;
  using component_list =
      tmpl::list<component<Dim, MetavariablesAnalyticSolution>>;
  using equation_of_state_tag = EquationOfStateTag;
  using system = System<Dim, HasPrimitives>;
  using analytic_variables_tags =
      tmpl::conditional_t<HasPrimitives,
                          typename system::primitive_variables_tag::tags_list,
                          typename system::variables_tag::tags_list>;
  using temporal_id = TimeId;
  using const_global_cache_tags =
      tmpl::list<Tags::AnalyticSolution<analytic_solution>>;
  enum class Phase { Initialization, Exit };
};

template <size_t Dim, bool HasPrimitives>
void test_analytic_solution() noexcept {
  using metavars = MetavariablesAnalyticSolution<Dim, HasPrimitives>;
  using comp = component<Dim, metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{{SystemAnalyticSolution{}}};
  const double initial_time = 1.3;
  const auto inertial_coords =
      emplace_component<Dim>(make_not_null(&runner), initial_time);
  Variables<tmpl::list<Var>> var(get<0>(inertial_coords).size(), 8.9999);
  Variables<tmpl::list<PrimVar>> prim_var(get<0>(inertial_coords).size(),
                                          9.9999);

  // Invoke the SetVariables action on the runner
  ActionTesting::next_action<comp>(make_not_null(&runner), 0);
  if (HasPrimitives) {
    prim_var.assign_subset(SystemAnalyticSolution{}.variables(
        inertial_coords, initial_time, tmpl::list<PrimVar>{}));
  } else {
    var.assign_subset(SystemAnalyticSolution{}.variables(
        inertial_coords, initial_time, tmpl::list<Var>{}));
  }
  CHECK(ActionTesting::get_databox_tag<comp, Var>(runner, 0) == get<Var>(var));
  CHECK(ActionTesting::get_databox_tag<comp, PrimVar>(runner, 0) ==
        get<PrimVar>(prim_var));
}

template <size_t Dim, bool HasPrimitives>
struct MetavariablesAnalyticData {
  static constexpr size_t volume_dim = Dim;
  using analytic_data = SystemAnalyticData;
  using component_list = tmpl::list<component<Dim, MetavariablesAnalyticData>>;
  using equation_of_state_tag = EquationOfStateTag;
  using system = System<Dim, HasPrimitives>;
  using analytic_variables_tags =
      tmpl::conditional_t<HasPrimitives,
                          typename system::primitive_variables_tag::tags_list,
                          typename system::variables_tag::tags_list>;
  using temporal_id = TimeId;
  using const_global_cache_tags = tmpl::list<Tags::AnalyticData<analytic_data>>;
  enum class Phase { Initialization, Exit };
};

template <size_t Dim, bool HasPrimitives>
void test_analytic_data() noexcept {
  using metavars = MetavariablesAnalyticData<Dim, HasPrimitives>;
  using comp = component<Dim, metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{{SystemAnalyticData{}}};
  const double initial_time = 1.3;
  const auto inertial_coords =
      emplace_component<Dim>(make_not_null(&runner), initial_time);
  Variables<tmpl::list<Var>> var(get<0>(inertial_coords).size(), 8.9999);
  Variables<tmpl::list<PrimVar>> prim_var(get<0>(inertial_coords).size(),
                                          9.9999);

  // Invoke the SetVariables action on the runner
  ActionTesting::next_action<comp>(make_not_null(&runner), 0);
  if (HasPrimitives) {
    prim_var.assign_subset(
        SystemAnalyticData{}.variables(inertial_coords, tmpl::list<PrimVar>{}));
  } else {
    var.assign_subset(
        SystemAnalyticData{}.variables(inertial_coords, tmpl::list<Var>{}));
  }
  CHECK(ActionTesting::get_databox_tag<comp, Var>(runner, 0) == get<Var>(var));
  CHECK(ActionTesting::get_databox_tag<comp, PrimVar>(runner, 0) ==
        get<PrimVar>(prim_var));
}

SPECTRE_TEST_CASE("Unit.Evolution.Initialization.SetVariables",
                  "[Unit][Evolution][Actions]") {
  domain::FunctionsOfTime::register_derived_with_charm();
  Parallel::register_classes_in_list<
      tmpl::list<domain::CoordinateMap<Frame::Logical, Frame::Grid,
                                       domain::CoordinateMaps::Identity<1>>,
                 domain::CoordinateMap<Frame::Logical, Frame::Grid,
                                       domain::CoordinateMaps::Identity<2>>,
                 domain::CoordinateMap<Frame::Logical, Frame::Grid,
                                       domain::CoordinateMaps::Identity<3>>,
                 domain::CoordinateMap<
                     Frame::Grid, Frame::Inertial,
                     domain::CoordinateMaps::TimeDependent::CubicScale<1>>,
                 domain::CoordinateMap<
                     Frame::Grid, Frame::Inertial,
                     domain::CoordinateMaps::TimeDependent::CubicScale<2>>,
                 domain::CoordinateMap<
                     Frame::Grid, Frame::Inertial,
                     domain::CoordinateMaps::TimeDependent::CubicScale<3>>>>();

  // Test setting variables from analytic solution
  test_analytic_solution<1, false>();
  test_analytic_solution<1, true>();
  test_analytic_solution<2, false>();
  test_analytic_solution<2, true>();
  test_analytic_solution<3, false>();
  test_analytic_solution<3, true>();

  // Test setting variables from analytic data
  test_analytic_data<1, false>();
  test_analytic_data<1, true>();
  test_analytic_data<2, false>();
  test_analytic_data<2, true>();
  test_analytic_data<3, false>();
  test_analytic_data<3, true>();
}
}  // namespace
