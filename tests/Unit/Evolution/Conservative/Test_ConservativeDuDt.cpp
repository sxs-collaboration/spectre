// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Conservative/ConservativeDuDt.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

// IWYU pragma: no_forward_declare Tags::Flux
// IWYU pragma: no_forward_declare Tags::div

namespace {
using frame = Frame::Inertial;
constexpr size_t volume_dim = 2;

struct Var1 : db::SimpleTag {
  static std::string name() noexcept { return "Var1"; }
  using type = Scalar<DataVector>;
};

struct Var2 : db::SimpleTag {
  static std::string name() noexcept { return "Var2"; }
  using type = tnsr::i<DataVector, volume_dim, frame>;
};

template <typename SourcedVariables>
struct System {
  using variables_tag = Tags::Variables<tmpl::list<Var1, Var2>>;
  using sourced_variables = SourcedVariables;
  static constexpr size_t volume_dim = ::volume_dim;
};

template <typename Var>
using divergence_flux =
    Tags::div<Tags::Flux<Var, tmpl::size_t<volume_dim>, frame>>;

using divergence_tags =
    tmpl::list<divergence_flux<Var1>, divergence_flux<Var2>>;

static_assert(
    cpp17::is_same_v<ConservativeDuDt<System<tmpl::list<>>>::argument_tags,
                     divergence_tags>,
    "Failed testing ConservativeDuDt::argument_tags");
static_assert(
    cpp17::is_same_v<ConservativeDuDt<System<tmpl::list<Var1>>>::argument_tags,
                     tmpl::push_back<divergence_tags, Tags::Source<Var1>>>,
    "Failed testing ConservativeDuDt::argument_tags");
static_assert(
    cpp17::is_same_v<ConservativeDuDt<System<tmpl::list<Var2>>>::argument_tags,
                     tmpl::push_back<divergence_tags, Tags::Source<Var2>>>,
    "Failed testing ConservativeDuDt::argument_tags");
static_assert(
    cpp17::is_same_v<
        ConservativeDuDt<System<tmpl::list<Var1, Var2>>>::argument_tags,
        tmpl::push_back<divergence_tags, Tags::Source<Var1>,
                        Tags::Source<Var2>>>,
    "Failed testing ConservativeDuDt::argument_tags");
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ConservativeDuDt", "[Unit][Evolution]") {
  constexpr size_t num_points = 5;

  const Scalar<DataVector> divflux_var1{{{{1., 2., 3., 4., 5.}}}};
  const tnsr::i<DataVector, volume_dim, frame> divflux_var2{
      {{{11., 12., 13., 14., 15.}, {21., 22., 23., 24., 25.}}}};

  const Scalar<DataVector> source_var1{{{{5., 4., 3., 2., 1.}}}};
  const tnsr::i<DataVector, volume_dim, frame> source_var2{
      {{{15., 14., 13., 12., 11.}, {25., 24., 23., 22., 21.}}}};

  Scalar<DataVector> dt_var1(num_points);
  tnsr::i<DataVector, volume_dim, frame> dt_var2(num_points);

  ConservativeDuDt<System<tmpl::list<>>>::apply(
      &dt_var1, &dt_var2, divflux_var1, divflux_var2);
  CHECK(get(dt_var1) == -get(divflux_var1));
  CHECK(get<0>(dt_var2) == -get<0>(divflux_var2));
  CHECK(get<1>(dt_var2) == -get<1>(divflux_var2));

  ConservativeDuDt<System<tmpl::list<Var1>>>::apply(
      &dt_var1, &dt_var2, divflux_var1, divflux_var2, source_var1);
  CHECK(get(dt_var1) == get(source_var1) - get(divflux_var1));
  CHECK(get<0>(dt_var2) == -get<0>(divflux_var2));
  CHECK(get<1>(dt_var2) == -get<1>(divflux_var2));

  ConservativeDuDt<System<tmpl::list<Var2>>>::apply(
      &dt_var1, &dt_var2, divflux_var1, divflux_var2, source_var2);
  CHECK(get(dt_var1) == -get(divflux_var1));
  CHECK(get<0>(dt_var2) == get<0>(source_var2) - get<0>(divflux_var2));
  CHECK(get<1>(dt_var2) == get<1>(source_var2) - get<1>(divflux_var2));

  ConservativeDuDt<System<tmpl::list<Var1, Var2>>>::apply(
      &dt_var1, &dt_var2, divflux_var1, divflux_var2, source_var1, source_var2);
  CHECK(get(dt_var1) == get(source_var1) - get(divflux_var1));
  CHECK(get<0>(dt_var2) == get<0>(source_var2) - get<0>(divflux_var2));
  CHECK(get<1>(dt_var2) == get<1>(source_var2) - get<1>(divflux_var2));
}
