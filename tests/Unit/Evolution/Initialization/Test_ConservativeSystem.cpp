// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/ConservativeSystem.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Phase.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() { return "Var"; }
};

struct PrimVar : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() { return "PrimVar"; }
};

struct SystemAnalyticSolution : public MarkAsAnalyticSolution {
  const EquationsOfState::PolytropicFluid<true> equation_of_state_{100.0, 2.0};
  const auto& equation_of_state() const { return equation_of_state_; }

  // NOLINTNEXTLINE
  void pup(PUP::er& /*p*/) {}
};

template <size_t Dim, bool HasPrimitiveAndConservativeVars>
struct System {
  static constexpr bool is_in_flux_conservative_form = true;
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
  using const_global_cache_tag_list =
      tmpl::list<typename Metavariables::equation_of_state_tag>;

  using initial_tags = tmpl::list<domain::Tags::Mesh<Dim>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<initial_tags>,
                 Initialization::Actions::ConservativeSystem<
                     typename Metavariables::system>>>>;
};

template <size_t Dim, bool HasPrimitives>
struct Metavariables {
  using component_list = tmpl::list<component<Dim, Metavariables>>;
  using system = System<Dim, HasPrimitives>;
  using const_global_cache_tags =
      tmpl::list<Tags::AnalyticSolution<SystemAnalyticSolution>>;

  struct equation_of_state_tag : db::SimpleTag {
    using type = std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>;
  };
};

template <size_t Dim, typename Runner>
void check_primitives(std::true_type /*has_prims*/, const Runner& runner,
                      const size_t number_of_grid_points) {
  using metavars = Metavariables<Dim, true>;
  using comp = component<Dim, metavars>;
  using prim_vars_tag = Tags::Variables<tmpl::list<PrimVar>>;
  SystemAnalyticSolution system_analytic_solution{};
  CHECK(ActionTesting::get_databox_tag<comp, prim_vars_tag>(runner, 0)
            .number_of_grid_points() == number_of_grid_points);
}

template <size_t Dim, typename Runner>
void check_primitives(std::false_type /*has_prims*/, const Runner& /*runner*/,
                      const size_t /*number_of_grid_points*/) {}

template <size_t Dim, bool HasPrimitives>
void test() {
  using metavars = Metavariables<Dim, HasPrimitives>;
  using comp = component<Dim, metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{{SystemAnalyticSolution{}}};
  Mesh<Dim> mesh{5, SpatialDiscretization::Basis::Legendre,
                 SpatialDiscretization::Quadrature::GaussLobatto};
  ActionTesting::emplace_component_and_initialize<comp>(&runner, 0, {mesh});
  // Invoke the ConservativeSystem action on the runner
  ActionTesting::next_action<comp>(make_not_null(&runner), 0);
  using vars_tag = Tags::Variables<tmpl::list<Var>>;
  // The numerical value that the vars are set to is undefined, but the number
  // of grid points must be correct.
  CHECK(ActionTesting::get_databox_tag<comp, vars_tag>(runner, 0)
            .number_of_grid_points() == mesh.number_of_grid_points());
  check_primitives<Dim>(std::integral_constant<bool, HasPrimitives>{}, runner,
                        mesh.number_of_grid_points());
}

SPECTRE_TEST_CASE("Unit.Evolution.Initialization.ConservativeSystem",
                  "[Unit][Evolution][Actions]") {
  test<1, true>();
  test<2, true>();
  test<3, true>();
  test<1, false>();
  test<2, false>();
  test<3, false>();
}
}  // namespace
