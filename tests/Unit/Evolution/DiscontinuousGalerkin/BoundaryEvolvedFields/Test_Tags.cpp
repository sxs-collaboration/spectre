// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryEvolvedFields/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// Interior source fields whose boundary twins we evolve.
struct Psi : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct Pi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

using evolution::dg::BoundaryEvolvedFields::bc_opts_in_v;
using evolution::dg::BoundaryEvolvedFields::boundary_evolved_field_tags;
using evolution::dg::BoundaryEvolvedFields::
    boundary_evolved_fields_are_homogeneous_v;
using evolution::dg::BoundaryEvolvedFields::boundary_evolved_variables_of;
using evolution::dg::Tags::BoundaryValue;

using field_tags = tmpl::list<BoundaryValue<Psi>>;
template <size_t Dim>
using values_tag =
    evolution::dg::Tags::BoundaryEvolvedFieldsValues<Dim, field_tags>;
template <size_t Dim>
using dt_stash_tag =
    evolution::dg::Tags::BoundaryEvolvedFieldsDtStash<Dim, field_tags>;
template <size_t Dim>
using history_tag =
    evolution::dg::Tags::BoundaryEvolvedFieldsHistory<Dim, field_tags>;

void test_tag_names() {
  // A prefix tag: the composed name disambiguates the wrapped source field.
  TestHelpers::db::test_prefix_tag<BoundaryValue<Psi>>("BoundaryValue(Psi)");
  TestHelpers::db::test_simple_tag<values_tag<1>>(
      "BoundaryEvolvedFieldsValues");
  TestHelpers::db::test_simple_tag<dt_stash_tag<1>>(
      "BoundaryEvolvedFieldsDtStash");
  TestHelpers::db::test_simple_tag<history_tag<1>>(
      "BoundaryEvolvedFieldsHistory");
}

struct MockBcPsi {
  using boundary_evolved_variables = tmpl::list<BoundaryValue<Psi>>;
};
// A second boundary condition declaring the identical field set, for the
// homogeneous-multi-boundary-condition and de-duplicate checks.
struct MockBcPsiAgain {
  using boundary_evolved_variables = tmpl::list<BoundaryValue<Psi>>;
};
// A heterogeneous declaration (a different field set); forbidden by the
// homogeneity contract, used only for the negative homogeneity check.
struct MockBcPsiPi {
  using boundary_evolved_variables =
      tmpl::list<BoundaryValue<Psi>, BoundaryValue<Pi>>;
};
// Declares nothing: it must compile away (contribute no field tags).
struct MockBcNonOpting {};

void test_union_metafunction() {
  INFO("boundary_evolved_field_tags union, opt-in trait, homogeneity guard");
  static_assert(not bc_opts_in_v<MockBcNonOpting>,
                "A boundary condition that declares nothing must not opt in.");
  static_assert(bc_opts_in_v<MockBcPsi>);
  static_assert(bc_opts_in_v<MockBcPsiPi>);
  static_assert(std::is_same_v<boundary_evolved_variables_of<MockBcNonOpting>,
                               tmpl::list<>>);

  // The union is the flat, duplicate-free set over the derived boundary
  // conditions: a multi-field condition keeps its fields, a non-opting one
  // drops out, and two conditions declaring the same set collapse to one.
  static_assert(
      std::is_same_v<boundary_evolved_field_tags<
                         tmpl::list<MockBcPsiPi, MockBcPsi, MockBcNonOpting>>,
                     tmpl::list<BoundaryValue<Psi>, BoundaryValue<Pi>>>);
  static_assert(std::is_same_v<boundary_evolved_field_tags<tmpl::list<
                                   MockBcPsi, MockBcPsiAgain, MockBcNonOpting>>,
                               tmpl::list<BoundaryValue<Psi>>>);
  static_assert(
      std::is_same_v<boundary_evolved_field_tags<tmpl::list<MockBcNonOpting>>,
                     tmpl::list<>>);

  // Homogeneity guard: every opting boundary condition must declare the
  // identical field set; a heterogeneous pair is rejected.
  static_assert(boundary_evolved_fields_are_homogeneous_v<
                tmpl::list<MockBcPsi, MockBcPsiAgain, MockBcNonOpting>>);
  static_assert(not boundary_evolved_fields_are_homogeneous_v<
                tmpl::list<MockBcPsi, MockBcPsiPi>>);
}

// Checkpoint/restart: the per-face storage tags are ordinary mutable DataBox
// SimpleTags (invisible to `get_all_history_tags`, but serialized by the
// DataBox like any other), so the boundary state must round-trip through
// serialization.
void test_serialization() {
  INFO("The facility storage tags round-trip through serialization");
  constexpr size_t Dim = 1;
  const auto direction = Direction<Dim>::lower_xi();
  const size_t number_of_pts = 3;
  const size_t order = 2;
  const Slab slab(0., 1.);

  Variables<field_tags> face_value{number_of_pts};
  get(get<BoundaryValue<Psi>>(face_value)) = DataVector{number_of_pts, 1.5};
  typename dt_stash_tag<Dim>::type::mapped_type face_deriv{number_of_pts};
  get(get<Tags::dt<BoundaryValue<Psi>>>(face_deriv)) =
      DataVector{number_of_pts, 2.5};

  typename values_tag<Dim>::type values{};
  values.insert({direction, face_value});
  typename dt_stash_tag<Dim>::type dt_stash{};
  dt_stash.insert({direction, face_deriv});
  typename history_tag<Dim>::type histories{};
  typename history_tag<Dim>::type::mapped_type history{order};
  history.insert(TimeStepId(true, 0, slab.start()), face_value, face_deriv);
  histories.insert({direction, std::move(history)});

  CHECK(serialize_and_deserialize(values) == values);
  CHECK(serialize_and_deserialize(dt_stash) == dt_stash);
  // `TimeSteppers::History` has no `operator==`; verify the round-tripped
  // per-face history keeps its order, its records, and their contents.
  const auto histories2 = serialize_and_deserialize(histories);
  const auto& h = histories2.at(direction);
  CHECK(h.integration_order() == order);
  CHECK(h.size() == 1);
  CHECK(h.back().time_step_id == TimeStepId(true, 0, slab.start()));
  CHECK(get(get<BoundaryValue<Psi>>(*h.back().value)) ==
        get(get<BoundaryValue<Psi>>(face_value)));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Dg.BoundaryEvolvedFields.Tags",
                  "[Unit][Evolution]") {
  test_tag_names();
  test_union_metafunction();
  test_serialization();
}
