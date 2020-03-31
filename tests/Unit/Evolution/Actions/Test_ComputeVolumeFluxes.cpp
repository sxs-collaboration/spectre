// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Actions/ComputeVolumeFluxes.hpp"  // IWYU pragma: keep
#include "Framework/ActionTesting.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <unordered_map>

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox
// IWYU pragma: no_forward_declare db::DataBox
// IWYU pragma: no_forward_declare Tensor

namespace {
constexpr size_t dim = 2;

struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct Var2 : db::SimpleTag {
  using type = tnsr::I<DataVector, dim, Frame::Inertial>;
};

using flux_tag =
    db::add_tag_prefix<::Tags::Flux, Tags::Variables<tmpl::list<Var1>>,
                       tmpl::size_t<dim>, Frame::Inertial>;

struct ComputeFluxes {
  using argument_tags = tmpl::list<Var2, Var1>;
  using return_tags =
      tmpl::list<::Tags::Flux<Var1, tmpl::size_t<dim>, Frame::Inertial>>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, dim, Frame::Inertial>*> flux1,
      const tnsr::I<DataVector, dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var1) noexcept {
    get<0>(*flux1) = get(var1) * (get<0>(var2) - get<1>(var2));
    get<1>(*flux1) = get(var1) * (get<0>(var2) + get<1>(var2));
  }
};

struct System {
  using variables_tag = Tags::Variables<tmpl::list<Var1>>;
  using volume_fluxes = ComputeFluxes;
};

using ElementIdType = ElementId<dim>;

template <typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIdType;
  using simple_tags = tmpl::list<Tags::Variables<tmpl::list<Var1>>, Var2,
                                 flux_tag, domain::Tags::MeshVelocity<dim>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing,
                             tmpl::list<Actions::ComputeVolumeFluxes>>>;
};

struct Metavariables {
  static constexpr size_t volume_dim = dim;
  using component_list = tmpl::list<component<Metavariables>>;
  using system = System;
  enum class Phase { Initialization, Testing, Exit };
};

template <bool UseMovingMesh>
void test() noexcept {
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  const double var1_value = 3.;
  const ElementId<dim> self_id(1);
  MockRuntimeSystem runner{{}};

  boost::optional<tnsr::I<DataVector, dim, Frame::Inertial>> mesh_velocity{};
  if (UseMovingMesh) {
    mesh_velocity =
        tnsr::I<DataVector, dim, Frame::Inertial>{{{{1.2}, {-1.4}}}};
  }
  ActionTesting::emplace_component_and_initialize<component<Metavariables>>(
      &runner, self_id,
      {Variables<tmpl::list<Var1>>{1, var1_value},
       tnsr::I<DataVector, dim, Frame::Inertial>{{{{7.}, {12.}}}},
       db::item_type<flux_tag>{1, -100.}, mesh_velocity});
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);
  runner.next_action<component<Metavariables>>(self_id);

  tnsr::I<DataVector, dim, Frame::Inertial> expected_flux{{{{-15.}, {57.}}}};
  if (UseMovingMesh) {
    get<0>(expected_flux) -= get<0>(*mesh_velocity) * var1_value;
    get<1>(expected_flux) -= get<1>(*mesh_velocity) * var1_value;
  }
  CHECK(ActionTesting::get_databox_tag<
            component<Metavariables>,
            Tags::Flux<Var1, tmpl::size_t<dim>, Frame::Inertial>>(
            runner, self_id) == expected_flux);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ComputeVolumeFluxes",
                  "[Unit][Evolution][Actions]") {
  test<false>();
  test<true>();
}
