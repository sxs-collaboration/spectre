// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Evolution/Actions/ComputeVolumeFluxes.hpp"  // IWYU pragma: keep
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <unordered_map>

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox
// IWYU pragma: no_forward_declare db::DataBox
// IWYU pragma: no_forward_declare Tensor

namespace {
constexpr size_t dim = 2;

struct Var1 : db::SimpleTag {
  using type = Scalar<double>;
};

struct Var2 : db::SimpleTag {
  using type = tnsr::I<double, dim, Frame::Inertial>;
};

using flux_tag = Tags::Flux<Var1, tmpl::size_t<dim>, Frame::Inertial>;

struct ComputeFluxes {
  using argument_tags = tmpl::list<Var2, Var1>;
  using return_tags = tmpl::list<flux_tag>;
  static void apply(
      const gsl::not_null<tnsr::I<double, dim, Frame::Inertial>*> flux1,
      const tnsr::I<double, dim, Frame::Inertial>& var2,
      const Scalar<double>& var1) noexcept {
    get<0>(*flux1) = get(var1) * (get<0>(var2) - get<1>(var2));
    get<1>(*flux1) = get(var1) * (get<0>(var2) + get<1>(var2));
  }
};

struct System {
  using variables_tag = Var1;
  using volume_fluxes = ComputeFluxes;
};

using ElementIndexType = ElementIndex<dim>;

template <typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndexType;
  using simple_tags = tmpl::list<Var1, Var2, flux_tag>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing,
                             tmpl::list<Actions::ComputeVolumeFluxes>>>;
};

struct Metavariables {
  using component_list = tmpl::list<component<Metavariables>>;
  using system = System;
  enum class Phase { Initialization, Testing, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ComputeVolumeFluxes",
                  "[Unit][Evolution][Actions]") {
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;

  const ElementId<dim> self_id(1);

  using simple_tags = db::AddSimpleTags<Var1, Var2, flux_tag>;
  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_component_and_initialize<component<Metavariables>>(
      &runner, self_id,
      {db::item_type<Var1>{{{3.}}}, db::item_type<Var2>{{{7., 12.}}},
       db::item_type<flux_tag>{{{-100.}}}});
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);
  runner.next_action<component<Metavariables>>(self_id);

  auto& box = ActionTesting::get_databox<component<Metavariables>, simple_tags>(
      runner, self_id);
  CHECK(get<0>(db::get<flux_tag>(box)) == -15.);
  CHECK(get<1>(db::get<flux_tag>(box)) == 57.);
}
