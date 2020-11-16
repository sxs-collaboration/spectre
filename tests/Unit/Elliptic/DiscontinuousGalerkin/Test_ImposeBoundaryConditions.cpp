// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeBoundaryConditions.hpp"  // IWYU pragma: keep
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// Note: Most of this test is adapted from:
// `NumericalAlgorithms/DiscontinuousGalerkin/Actions/
// Test_ImposeBoundaryConditions.cpp`

namespace {
constexpr size_t Dim = 2;

struct ScalarField : db::SimpleTag {
  using type = Scalar<DataVector>;
};

using field_tag = ScalarField;
using vars_tag = Tags::Variables<tmpl::list<field_tag>>;

using interior_bdry_vars_tag =
    domain::Tags::Interface<domain::Tags::BoundaryDirectionsInterior<Dim>,
                            vars_tag>;
using exterior_bdry_vars_tag =
    domain::Tags::Interface<domain::Tags::BoundaryDirectionsExterior<Dim>,
                            vars_tag>;

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using const_global_cache_tags = tmpl::list<>;

  using simple_tags =
      db::AddSimpleTags<interior_bdry_vars_tag, exterior_bdry_vars_tag>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<elliptic::dg::Actions::
                         ImposeHomogeneousDirichletBoundaryConditions<
                             vars_tag, tmpl::list<field_tag>>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.Elliptic.DG.Actions.BoundaryConditions",
                  "[Unit][Elliptic][Actions]") {
  using my_component = ElementArray<Metavariables>;
  // Just making up two "external" directions and an element id
  const auto external_directions = {Direction<2>::lower_eta(),
                                    Direction<2>::upper_xi()};
  const ElementId<Dim> self_id{0};
  const size_t num_points = 3;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  {
    typename interior_bdry_vars_tag::type interior_bdry_vars;
    typename exterior_bdry_vars_tag::type exterior_bdry_vars;
    for (const auto& direction : external_directions) {
      interior_bdry_vars[direction].initialize(3);
      exterior_bdry_vars[direction].initialize(3);
    }
    get<field_tag>(interior_bdry_vars[Direction<2>::lower_eta()]) =
        Scalar<DataVector>{num_points, 1.};
    get<field_tag>(interior_bdry_vars[Direction<2>::upper_xi()]) =
        Scalar<DataVector>{num_points, 2.};

    ActionTesting::emplace_component_and_initialize<my_component>(
        &runner, self_id,
        {std::move(interior_bdry_vars), std::move(exterior_bdry_vars)});
  }
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);

  ActionTesting::next_action<my_component>(make_not_null(&runner), self_id);

  // Check that BC's were indeed applied.
  const auto& exterior_vars =
      ActionTesting::get_databox_tag<my_component, exterior_bdry_vars_tag>(
          runner, self_id);
  typename exterior_bdry_vars_tag::type expected_vars{};
  for (const auto& direction : external_directions) {
    expected_vars[direction].initialize(3);
  }
  get<field_tag>(expected_vars[Direction<2>::lower_eta()]) =
      Scalar<DataVector>{num_points, -1.};
  get<field_tag>(expected_vars[Direction<2>::upper_xi()]) =
      Scalar<DataVector>{num_points, -2.};
  CHECK(exterior_vars == expected_vars);
}

}  // namespace
