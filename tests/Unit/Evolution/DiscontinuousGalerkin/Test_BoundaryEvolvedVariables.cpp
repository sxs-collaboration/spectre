// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/BoundaryVariablesTag.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryEvolvedVariables.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/TestTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using ScalarTag = TestHelpers::Tags::Scalar<DataVector>;
using Scalar2Tag = TestHelpers::Tags::Scalar2<DataVector>;
using BoundaryScalar = evolution::dg::Tags::BoundaryValue<ScalarTag>;

static_assert(std::is_same_v<BoundaryScalar::type, ScalarTag::type>);
static_assert(std::is_same_v<BoundaryScalar::tag, ScalarTag>);

// Systems for the boundary-variables detection: split with a
// BoundaryVariables entry, split without one, and single-tag.
struct SplitSystem {
  using variables_tag =
      tmpl::list<::Tags::Variables<tmpl::list<ScalarTag>>,
                 ::Tags::BoundaryVariables<2, tmpl::list<BoundaryScalar>>>;
};
struct SplitWithoutBoundarySystem {
  using variables_tag = tmpl::list<::Tags::Variables<tmpl::list<ScalarTag>>,
                                   ::Tags::Variables<tmpl::list<Scalar2Tag>>>;
};
struct SingleTagSystem {
  using variables_tag = ::Tags::Variables<tmpl::list<ScalarTag>>;
};

static_assert(evolution::dg::system_has_boundary_variables_v<SplitSystem>);
static_assert(not evolution::dg::system_has_boundary_variables_v<
              SplitWithoutBoundarySystem>);
static_assert(
    not evolution::dg::system_has_boundary_variables_v<SingleTagSystem>);
static_assert(
    std::is_same_v<evolution::dg::boundary_variables_tag<SplitSystem>,
                   ::Tags::BoundaryVariables<2, tmpl::list<BoundaryScalar>>>);
static_assert(
    std::is_same_v<evolution::dg::boundary_variables_tag<SingleTagSystem>,
                   NoSuchType>);

struct OptingCondition {
  static constexpr bool evolves_boundary_variables = true;
  using boundary_field_time_derivatives_evolved_variables_tags =
      tmpl::list<ScalarTag>;
  using boundary_field_time_derivatives_temporary_tags = tmpl::list<Scalar2Tag>;
  static std::optional<std::string> boundary_field_time_derivatives(
      gsl::not_null<Scalar<DataVector>*> /*dt_boundary_scalar*/) {
    return std::nullopt;
  }
};
struct NonOptingCondition {
  static std::optional<std::string> dg_ghost() { return std::nullopt; }
};
struct MethodWithoutMarkerCondition {
  static std::optional<std::string> boundary_field_time_derivatives(
      gsl::not_null<Scalar<DataVector>*> /*dt_boundary_scalar*/) {
    return std::nullopt;
  }
};
struct MarkerWithoutMethodCondition {
  static constexpr bool evolves_boundary_variables = true;
};

static_assert(evolution::dg::evolves_boundary_variables_v<OptingCondition>);
static_assert(
    not evolution::dg::evolves_boundary_variables_v<NonOptingCondition>);
static_assert(not evolution::dg::evolves_boundary_variables_v<
              MethodWithoutMarkerCondition>);
static_assert(
    evolution::dg::evolves_boundary_variables_v<MarkerWithoutMethodCondition>);

static_assert(evolution::dg::detail::has_boundary_field_time_derivatives_v<
              OptingCondition>);
static_assert(evolution::dg::detail::has_boundary_field_time_derivatives_v<
              MethodWithoutMarkerCondition>);
static_assert(not evolution::dg::detail::has_boundary_field_time_derivatives_v<
              NonOptingCondition>);
static_assert(not evolution::dg::detail::has_boundary_field_time_derivatives_v<
              MarkerWithoutMethodCondition>);

// The assembled interior inputs are ordered evolved, primitive, temporary,
// with each undeclared list defaulting to empty.
static_assert(
    std::is_same_v<evolution::dg::boundary_field_time_derivatives_interior_tags<
                       OptingCondition>,
                   tmpl::list<ScalarTag, Scalar2Tag>>);
static_assert(
    std::is_same_v<evolution::dg::boundary_field_time_derivatives_interior_tags<
                       NonOptingCondition>,
                   tmpl::list<>>);

SPECTRE_TEST_CASE("Unit.Evolution.Dg.BoundaryEvolvedVariables",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_prefix_tag<BoundaryScalar>("BoundaryValue(Scalar)");
}
}  // namespace
