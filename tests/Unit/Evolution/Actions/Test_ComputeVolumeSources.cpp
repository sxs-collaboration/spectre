// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Evolution/Actions/ComputeVolumeSources.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {
constexpr size_t dim = 2;

struct Var1 : db::SimpleTag {
  static std::string name() noexcept { return "Var1"; }
  using type = Scalar<DataVector>;
};

struct Var2 : db::SimpleTag {
  static std::string name() noexcept { return "Var2"; }
  using type = tnsr::I<DataVector, dim, Frame::Inertial>;
};

struct Var3 : db::SimpleTag {
  static std::string name() noexcept { return "Var3"; }
  using type = Scalar<DataVector>;
};

struct ComputeSources {
  using argument_tags = tmpl::list<Var1, Var3>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, dim, Frame::Inertial>*> source2,
      const Scalar<DataVector>& var1, const Scalar<DataVector>& var3) noexcept {
    get<0>(*source2) = get(var1);
    get<1>(*source2) = get(var3);
  }
};

struct System {
  static constexpr size_t volume_dim = dim;
  using variables_tag = Tags::Variables<tmpl::list<Var1, Var2>>;
  using sourced_variables = tmpl::list<Var2>;
  using volume_sources = ComputeSources;
};

using ElementIndexType = ElementIndex<dim>;

struct Metavariables;
using component = ActionTesting::MockArrayComponent<
    Metavariables, ElementIndexType, tmpl::list<>,
    tmpl::list<Actions::ComputeVolumeSources>>;

struct Metavariables {
  using component_list = tmpl::list<component>;
  using system = System;
  using const_global_cache_tag_list = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ComputeVolumeSources",
                  "[Unit][Evolution][Actions]") {
  ActionTesting::ActionRunner<Metavariables> runner{{}};
  const ElementId<dim> self(1);

  const Scalar<DataVector> var1{{{{3., 4.}}}};
  const Scalar<DataVector> var3{{{{5., 6.}}}};

  db::item_type<System::variables_tag> vars(2);
  get<Var1>(vars) = var1;

  using source_tag =
      Tags::Source<Tags::Variables<tmpl::list<Tags::Source<Var2>>>>;

  auto start_box =
      db::create<db::AddSimpleTags<System::variables_tag, Var3, source_tag>>(
          std::move(vars), var3, db::item_type<source_tag>(2));

  const auto box = get<0>(
      runner.apply<component, Actions::ComputeVolumeSources>(start_box, self));
  CHECK(get<0>(db::get<Tags::Source<Var2>>(box)) == get(var1));
  CHECK(get<1>(db::get<Tags::Source<Var2>>(box)) == get(var3));
}
