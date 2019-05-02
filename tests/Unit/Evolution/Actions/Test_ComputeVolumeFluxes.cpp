// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Evolution/Actions/ComputeVolumeFluxes.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <unordered_map>

// IWYU pragma: no_forward_declare db::DataBox
// IWYU pragma: no_forward_declare Tensor

namespace {
constexpr size_t dim = 2;

struct Var1 : db::SimpleTag {
  static std::string name() noexcept { return "Var1"; }
  using type = Scalar<double>;
};

struct Var2 : db::SimpleTag {
  static std::string name() noexcept { return "Var2"; }
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
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<Actions::ComputeVolumeFluxes>;
  using initial_databox =
      db::compute_databox_type<tmpl::list<Var1, Var2, flux_tag>>;
};

struct Metavariables {
  using component_list = tmpl::list<component<Metavariables>>;
  using system = System;
  using const_global_cache_tag_list = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ComputeVolumeFluxes",
                  "[Unit][Evolution][Actions]") {
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using MockDistributedObjectsTag =
      MockRuntimeSystem::MockDistributedObjectsTag<component<Metavariables>>;

  const ElementId<dim> self_id(1);

  using simple_tags = db::AddSimpleTags<Var1, Var2, flux_tag>;
  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(self_id,
               db::create<simple_tags>(db::item_type<Var1>{{{3.}}},
                                       db::item_type<Var2>{{{7., 12.}}},
                                       db::item_type<flux_tag>{{{-100.}}}));
  MockRuntimeSystem runner{{}, std::move(dist_objects)};

  runner.next_action<component<Metavariables>>(self_id);

  auto& box = runner.algorithms<component<Metavariables>>()
                  .at(self_id)
                  .get_databox<db::compute_databox_type<simple_tags>>();
  CHECK(get<0>(db::get<flux_tag>(box)) == -15.);
  CHECK(get<1>(db::get<flux_tag>(box)) == 57.);
}
