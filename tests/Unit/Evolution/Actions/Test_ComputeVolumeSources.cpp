// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Evolution/Actions/ComputeVolumeSources.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
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

using source_tag = ::Tags::Source<Var2>;

struct ComputeSources {
  using argument_tags = tmpl::list<Var1, Var3>;
  using return_tags = tmpl::list<source_tag>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, dim, Frame::Inertial>*> source2,
      const Scalar<DataVector>& var1, const Scalar<DataVector>& var3) noexcept {
    get<0>(*source2) = get(var1);
    get<1>(*source2) = get(var3);
  }
};

struct System {
  using variables_tag = Tags::Variables<tmpl::list<Var1, Var2>>;
  using sourced_variables = tmpl::list<Var2>;
  using volume_sources = ComputeSources;
};

using ElementIndexType = ElementIndex<dim>;


template <typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndexType;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<Actions::ComputeVolumeSources>;
  using initial_databox = db::compute_databox_type<
      tmpl::list<System::variables_tag, Var3, source_tag>>;
};

struct Metavariables {
  using component_list = tmpl::list<component<Metavariables>>;
  using system = System;
  using const_global_cache_tag_list = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ComputeVolumeSources",
                  "[Unit][Evolution][Actions]") {
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using MockDistributedObjectsTag =
      MockRuntimeSystem::MockDistributedObjectsTag<component<Metavariables>>;

  const ElementId<dim> self_id(1);
  const Scalar<DataVector> var1{{{{3., 4.}}}};
  const Scalar<DataVector> var3{{{{5., 6.}}}};

  db::item_type<System::variables_tag> vars(2);
  get<Var1>(vars) = var1;

  using simple_tags =
      db::AddSimpleTags<System::variables_tag, Var3, source_tag>;
  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(self_id,
               db::create<simple_tags>(std::move(vars), var3,
                                       db::item_type<source_tag>{2_st}));
  MockRuntimeSystem runner{{}, std::move(dist_objects)};

  runner.next_action<component<Metavariables>>(self_id);

  const auto& box = runner.algorithms<component<Metavariables>>()
                        .at(self_id)
                        .get_databox<db::compute_databox_type<simple_tags>>();
  CHECK(get<0>(db::get<source_tag>(box)) == get(var1));
  CHECK(get<1>(db::get<source_tag>(box)) == get(var3));
}
