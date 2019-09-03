// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace {

template <typename Tag>
struct Step : db::SimpleTag, db::PrefixTag {
  using type = typename Tag::type;
  using tag = Tag;
};

struct SomeField : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct AnotherField : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
};

template <size_t Dim>
void test_first_order_operator(const DataVector& used_for_size) {
  using vars_tag = Tags::Variables<tmpl::list<SomeField, AnotherField<Dim>>>;
  using step_tag = db::add_tag_prefix<Step, vars_tag>;
  using fluxes_tag = db::add_tag_prefix<::Tags::Flux, vars_tag,
                                        tmpl::size_t<Dim>, Frame::Inertial>;
  using div_fluxes_tag = db::add_tag_prefix<::Tags::div, fluxes_tag>;
  using sources_tag = db::add_tag_prefix<::Tags::Source, vars_tag>;
  using first_order_operator =
      elliptic::FirstOrderOperator<Dim, Step, vars_tag>;

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);

  const auto div_fluxes =
      make_with_random_values<db::item_type<div_fluxes_tag>>(
          nn_generator, nn_dist, used_for_size);
  const auto sources = make_with_random_values<db::item_type<sources_tag>>(
      nn_generator, nn_dist, used_for_size);
  const auto step = make_with_value<db::item_type<step_tag>>(
      used_for_size, std::numeric_limits<double>::signaling_NaN());

  auto box =
      db::create<db::AddSimpleTags<div_fluxes_tag, sources_tag, step_tag>>(
          div_fluxes, sources, step);

  db::mutate_apply<first_order_operator>(make_not_null(&box));

  const auto& computed_step = get<step_tag>(box);
  CHECK(get(get<Step<SomeField>>(computed_step)) ==
        -get(get<::Tags::div<
                 ::Tags::Flux<SomeField, tmpl::size_t<Dim>, Frame::Inertial>>>(
            div_fluxes)) +
            get(get<::Tags::Source<SomeField>>(sources)));
  for (size_t d = 0; d < Dim; d++) {
    CHECK(get<Step<AnotherField<Dim>>>(computed_step).get(d) ==
          -get<::Tags::div<::Tags::Flux<AnotherField<Dim>, tmpl::size_t<Dim>,
                                        Frame::Inertial>>>(div_fluxes)
                  .get(d) +
              get<::Tags::Source<AnotherField<Dim>>>(sources).get(d));
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.FirstOrderOperator", "[Unit][Elliptic]") {
  const DataVector used_for_size{5};
  test_first_order_operator<1>(used_for_size);
  test_first_order_operator<2>(used_for_size);
  test_first_order_operator<3>(used_for_size);
}
