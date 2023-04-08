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
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct TimeId : db::SimpleTag {
  using type = double;
};

struct Var : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct NonConservativeVar : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct PrimVar : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct EquationOfStateTag : db::SimpleTag {
  using type = std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>;
};

struct SystemAnalyticSolution : public MarkAsAnalyticSolution,
                                public evolution::initial_data::InitialData {
  SystemAnalyticSolution() = default;
  ~SystemAnalyticSolution() override = default;

  explicit SystemAnalyticSolution(CkMigrateMessage* msg)
      : evolution::initial_data::InitialData(msg) {}
  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(SystemAnalyticSolution);
#pragma GCC diagnostic pop

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override {
    return std::make_unique<SystemAnalyticSolution>(*this);
  }

  template <size_t Dim>
  tuples::TaggedTuple<Var, NonConservativeVar> variables(
      const tnsr::I<DataVector, Dim>& x, const double t,
      tmpl::list<Var, NonConservativeVar> /*meta*/) const {
    tuples::TaggedTuple<Var, NonConservativeVar> vars(x.get(0), x.get(0));
    for (size_t d = 1; d < Dim; ++d) {
      get(get<Var>(vars)) += square(x.get(d)) + t;
      get(get<NonConservativeVar>(vars)) += square(x.get(d)) / 5.0 - t;
    }
    return vars;
  }

  template <size_t Dim>
  tuples::TaggedTuple<Var> variables(const tnsr::I<DataVector, Dim>& x,
                                     const double t,
                                     tmpl::list<Var> /*meta*/) const {
    tuples::TaggedTuple<Var> vars(x.get(0) + t);
    for (size_t d = 1; d < Dim; ++d) {
      get(get<Var>(vars)) += x.get(d) + t;
    }
    return vars;
  }

  template <size_t Dim>
  tuples::TaggedTuple<NonConservativeVar> variables(
      const tnsr::I<DataVector, Dim>& x, const double t,
      tmpl::list<NonConservativeVar> /*meta*/) const {
    tuples::TaggedTuple<NonConservativeVar> vars(x.get(0));
    for (size_t d = 1; d < Dim; ++d) {
      get(get<NonConservativeVar>(vars)) += square(x.get(d)) / 5.0 - t;
    }
    return vars;
  }

  template <size_t Dim>
  tuples::TaggedTuple<PrimVar> variables(const tnsr::I<DataVector, Dim>& x,
                                         const double t,
                                         tmpl::list<PrimVar> /*meta*/) const {
    tuples::TaggedTuple<PrimVar> vars(2.0 * x.get(0) + t);
    for (size_t d = 1; d < Dim; ++d) {
      get(get<PrimVar>(vars)) += 2.0 * x.get(d) + t;
    }
    return vars;
  }

  // EoS just needs to be a dummy place holder
  static auto equation_of_state() {
    EquationsOfState::PolytropicFluid<true> equation_of_state_{100.0, 2.0};
    return equation_of_state_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    evolution::initial_data::InitialData::pup(p);
  }
};

PUP::able::PUP_ID SystemAnalyticSolution::my_PUP_ID = 0;

struct SystemAnalyticData : public MarkAsAnalyticData,
                            public evolution::initial_data::InitialData {
  SystemAnalyticData() = default;
  ~SystemAnalyticData() override = default;

  explicit SystemAnalyticData(CkMigrateMessage* msg)
      : evolution::initial_data::InitialData(msg) {}
  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(SystemAnalyticData);
#pragma GCC diagnostic pop

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override {
    return std::make_unique<SystemAnalyticData>(*this);
  }

  template <size_t Dim>
  tuples::TaggedTuple<Var, NonConservativeVar> variables(
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<Var, NonConservativeVar> /*meta*/) const {
    tuples::TaggedTuple<Var, NonConservativeVar> vars(x.get(0), x.get(0));
    for (size_t d = 1; d < Dim; ++d) {
      get(get<Var>(vars)) += square(x.get(d));
      get(get<NonConservativeVar>(vars)) += square(x.get(d)) / 5.0;
    }
    return vars;
  }

  template <size_t Dim>
  tuples::TaggedTuple<Var> variables(const tnsr::I<DataVector, Dim>& x,
                                     tmpl::list<Var> /*meta*/) const {
    tuples::TaggedTuple<Var> vars(x.get(0));
    for (size_t d = 1; d < Dim; ++d) {
      get(get<Var>(vars)) += square(x.get(d));
    }
    return vars;
  }

  template <size_t Dim>
  tuples::TaggedTuple<NonConservativeVar> variables(
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<NonConservativeVar> /*meta*/) const {
    tuples::TaggedTuple<NonConservativeVar> vars(x.get(0));
    for (size_t d = 1; d < Dim; ++d) {
      get(get<NonConservativeVar>(vars)) += square(x.get(d)) / 5.0;
    }
    return vars;
  }

  template <size_t Dim>
  tuples::TaggedTuple<PrimVar> variables(const tnsr::I<DataVector, Dim>& x,
                                         tmpl::list<PrimVar> /*meta*/) const {
    tuples::TaggedTuple<PrimVar> vars(2.0 * x.get(0));
    for (size_t d = 1; d < Dim; ++d) {
      get(get<PrimVar>(vars)) += square(2.0 * x.get(d));
    }
    return vars;
  }
  EquationsOfState::PolytropicFluid<true> equation_of_state_{100.0, 2.0};
  // EoS just needs to be a dummy place holder
  const auto& equation_of_state() { return equation_of_state_; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override { InitialData::pup(p); }
};

PUP::able::PUP_ID SystemAnalyticData::my_PUP_ID = 0;

template <size_t Dim, bool HasPrimitiveAndConservativeVars>
struct System {
  // is_in_flux_conservative_form is unused
  static constexpr bool is_in_flux_conservative_form = false;
  static constexpr bool has_primitive_and_conservative_vars =
      HasPrimitiveAndConservativeVars;
  using non_conservative_variables = tmpl::list<NonConservativeVar>;
  static constexpr size_t volume_dim = Dim;
  using variables_tag = Tags::Variables<tmpl::list<Var, NonConservativeVar>>;
  using primitive_variables_tag = Tags::Variables<tmpl::list<PrimVar>>;
};

template <size_t Dim, typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;

  using initial_tags =
      tmpl::list<Tags::Time,
                 domain::Tags::FunctionsOfTimeInitialize,
                 domain::Tags::Coordinates<Dim, Frame::ElementLogical>,
                 domain::Tags::ElementMap<Dim, Frame::Grid>,
                 domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                             Frame::Inertial>,
                 Tags::Variables<tmpl::list<Var, NonConservativeVar>>,
                 Tags::Variables<tmpl::list<PrimVar>>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<initial_tags>,
                 evolution::Initialization::Actions::SetVariables<
                     domain::Tags::Coordinates<Dim, Frame::ElementLogical>>>>>;
};

template <size_t Dim, typename Metavariables>
auto emplace_component(
    const gsl::not_null<ActionTesting::MockRuntimeSystem<Metavariables>*>
        runner,
    const double initial_time, const double expiration_time) {
  using comp = component<Dim, Metavariables>;

  const auto logical_coords = logical_coordinates(Mesh<Dim>{
      5, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto});
  ElementMap<Dim, Frame::Grid> logical_to_grid_map{
      ElementId<Dim>{0},
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
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
          initial_time, std::array<DataVector, 3>{{{1.0}, {-0.1}, {0.0}}},
          expiration_time)));
  Variables<tmpl::list<Var, NonConservativeVar>> var(
      get<0>(logical_coords).size(), 8.9999);
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

template <size_t Dim, bool HasPrimitives, bool UseInitialDataTag>
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
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<evolution::initial_data::InitialData,
                             tmpl::list<SystemAnalyticSolution>>>;
  };
  using const_global_cache_tags = tmpl::list<tmpl::conditional_t<
      UseInitialDataTag, evolution::initial_data::Tags::InitialData,
      Tags::AnalyticSolution<analytic_solution>>>;
};

template <size_t Dim, bool HasPrimitives, bool UseInitialDataTag>
void test_analytic_solution() {
  using metavars =
      MetavariablesAnalyticSolution<Dim, HasPrimitives, UseInitialDataTag>;
  using comp = component<Dim, metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner = []() {
    if constexpr (UseInitialDataTag) {
      return MockRuntimeSystem{
          {std::unique_ptr<evolution::initial_data::InitialData>(
              std::make_unique<SystemAnalyticSolution>())}};
    } else {
      return MockRuntimeSystem{{SystemAnalyticSolution{}}};
    }
  }();
  const double initial_time = 1.3;
  const double expiration_time = 2.5;
  const auto inertial_coords = emplace_component<Dim>(
      make_not_null(&runner), initial_time, expiration_time);
  Variables<tmpl::list<Var, NonConservativeVar>> var(
      get<0>(inertial_coords).size(), 8.9999);
  Variables<tmpl::list<PrimVar>> prim_var(get<0>(inertial_coords).size(),
                                          9.9999);

  // Invoke the SetVariables action on the runner
  ActionTesting::next_action<comp>(make_not_null(&runner), 0);
  if (HasPrimitives) {
    prim_var.assign_subset(SystemAnalyticSolution{}.variables(
        inertial_coords, initial_time, tmpl::list<PrimVar>{}));
    var.assign_subset(SystemAnalyticSolution{}.variables(
        inertial_coords, initial_time, tmpl::list<NonConservativeVar>{}));
  } else {
    var.assign_subset(SystemAnalyticSolution{}.variables(
        inertial_coords, initial_time, tmpl::list<Var, NonConservativeVar>{}));
  }
  CHECK(ActionTesting::get_databox_tag<comp, Var>(runner, 0) == get<Var>(var));
  CHECK(ActionTesting::get_databox_tag<comp, NonConservativeVar>(runner, 0) ==
        get<NonConservativeVar>(var));
  CHECK(ActionTesting::get_databox_tag<comp, PrimVar>(runner, 0) ==
        get<PrimVar>(prim_var));
}

template <size_t Dim, bool HasPrimitives, bool UseInitialDataTag>
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
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<evolution::initial_data::InitialData,
                             tmpl::list<SystemAnalyticData>>>;
  };
  using const_global_cache_tags =
      tmpl::list<tmpl::conditional_t<UseInitialDataTag,
                                     evolution::initial_data::Tags::InitialData,
                                     Tags::AnalyticData<analytic_data>>>;
};

template <size_t Dim, bool HasPrimitives, bool UseInitialDataTag>
void test_analytic_data() {
  using metavars =
      MetavariablesAnalyticData<Dim, HasPrimitives, UseInitialDataTag>;
  using comp = component<Dim, metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner = []() {
    if constexpr (UseInitialDataTag) {
      return MockRuntimeSystem{
          {std::unique_ptr<evolution::initial_data::InitialData>(
              std::make_unique<SystemAnalyticData>())}};
    } else {
      return MockRuntimeSystem{{SystemAnalyticData{}}};
    }
  }();
  const double initial_time = 1.3;
  const double expiration_time = 2.5;
  const auto inertial_coords = emplace_component<Dim>(
      make_not_null(&runner), initial_time, expiration_time);
  Variables<tmpl::list<Var, NonConservativeVar>> var(
      get<0>(inertial_coords).size(), 8.9999);
  Variables<tmpl::list<PrimVar>> prim_var(get<0>(inertial_coords).size(),
                                          9.9999);

  // Invoke the SetVariables action on the runner
  ActionTesting::next_action<comp>(make_not_null(&runner), 0);
  if (HasPrimitives) {
    prim_var.assign_subset(
        SystemAnalyticData{}.variables(inertial_coords, tmpl::list<PrimVar>{}));
    var.assign_subset(SystemAnalyticData{}.variables(
        inertial_coords, tmpl::list<NonConservativeVar>{}));
  } else {
    var.assign_subset(SystemAnalyticData{}.variables(
        inertial_coords, tmpl::list<Var, NonConservativeVar>{}));
  }
  CHECK(ActionTesting::get_databox_tag<comp, Var>(runner, 0) == get<Var>(var));
  CHECK(ActionTesting::get_databox_tag<comp, NonConservativeVar>(runner, 0) ==
        get<NonConservativeVar>(var));
  CHECK(ActionTesting::get_databox_tag<comp, PrimVar>(runner, 0) ==
        get<PrimVar>(prim_var));
}

template <size_t Dim, bool HasPrimitives>
void test_impl() {
  // Test setting variables from analytic solution
  test_analytic_solution<Dim, HasPrimitives, true>();
  test_analytic_solution<Dim, HasPrimitives, false>();
  // Test setting variables from analytic data
  test_analytic_data<Dim, HasPrimitives, true>();
  test_analytic_data<Dim, HasPrimitives, false>();
}

template <size_t Dim>
void test() {
  register_classes_with_charm<
      domain::CoordinateMap<Frame::BlockLogical, Frame::Grid,
                            domain::CoordinateMaps::Identity<Dim>>,
      domain::CoordinateMap<
          Frame::Grid, Frame::Inertial,
          domain::CoordinateMaps::TimeDependent::CubicScale<Dim>>>();

  test_impl<Dim, true>();
  test_impl<Dim, false>();
}

SPECTRE_TEST_CASE("Unit.Evolution.Initialization.SetVariables",
                  "[Unit][Evolution][Actions]") {
  domain::FunctionsOfTime::register_derived_with_charm();
  register_classes_with_charm<SystemAnalyticData, SystemAnalyticSolution>();
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
