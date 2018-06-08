// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Evolution/Actions/ComputeVolumeFluxes.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

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

struct ComputeFluxes {
  using argument_tags = tmpl::list<Var2, Var1>;
  static void apply(
      const gsl::not_null<tnsr::I<double, dim, Frame::Inertial>*> flux1,
      const tnsr::I<double, dim, Frame::Inertial>& var2,
      const Scalar<double>& var1) noexcept {
    get<0>(*flux1) = get(var1) * (get<0>(var2) - get<1>(var2));
    get<1>(*flux1) = get(var1) * (get<0>(var2) + get<1>(var2));
  }
};

struct System {
  static constexpr size_t volume_dim = dim;
  using variables_tag = Var1;
  using volume_fluxes = ComputeFluxes;
};

using ElementIndexType = ElementIndex<dim>;

struct Metavariables;
using component =
    ActionTesting::MockArrayComponent<Metavariables, ElementIndexType,
                                      tmpl::list<>,
                                      tmpl::list<Actions::ComputeVolumeFluxes>>;

struct Metavariables {
  using component_list = tmpl::list<component>;
  using system = System;
  using const_global_cache_tag_list = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ComputeVolumeFluxes",
                  "[Unit][Evolution][Actions]") {
  ActionTesting::ActionRunner<Metavariables> runner{{}};
  const ElementId<dim> self(1);

  using flux_tag = Tags::Flux<Var1, tmpl::size_t<dim>, Frame::Inertial>;
  auto start_box = db::create<db::AddSimpleTags<Var1, Var2, flux_tag>>(
      db::item_type<Var1>{{{3.}}}, db::item_type<Var2>{{{7., 12.}}},
      db::item_type<flux_tag>{{{-100.}}});

  const auto box = get<0>(
      runner.apply<component, Actions::ComputeVolumeFluxes>(start_box, self));
  CHECK(get<0>(db::get<flux_tag>(box)) == -15.);
  CHECK(get<1>(db::get<flux_tag>(box)) == 57.);
}
