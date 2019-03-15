// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Xcts/ConstantDensityStar.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

using field_tags = tmpl::list<Xcts::Tags::ConformalFactor<DataVector>>;
using auxiliary_field_tags = tmpl::list<
    Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial, DataVector>>;
using initial_tags =
    db::wrap_tags_in<Tags::Initial,
                     tmpl::append<field_tags, auxiliary_field_tags>>;
using source_tags =
    db::wrap_tags_in<Tags::Source,
                     tmpl::append<field_tags, auxiliary_field_tags>>;

struct ConstantDensityStarProxy : Xcts::Solutions::ConstantDensityStar {
  using Xcts::Solutions::ConstantDensityStar::ConstantDensityStar;
  tuples::tagged_tuple_from_typelist<field_tags> field_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
    return Xcts::Solutions::ConstantDensityStar::variables(x, field_tags{});
  }
  tuples::tagged_tuple_from_typelist<initial_tags> initial_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
    return Xcts::Solutions::ConstantDensityStar::variables(x, initial_tags{});
  }
  tuples::tagged_tuple_from_typelist<source_tags> source_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
    return Xcts::Solutions::ConstantDensityStar::variables(x, source_tags{});
  }
};

void test_solution(const double density, const double radius,
                   const std::string& options) {
  const ConstantDensityStarProxy solution(density, radius);
  const double test_radius = 2. * radius;
  pypp::check_with_random_values<1, field_tags>(
      &ConstantDensityStarProxy::field_variables, solution,
      "ConstantDensityStar", {"conformal_factor", "conformal_factor_gradient"},
      {{{-test_radius, test_radius}}}, std::make_tuple(density, radius),
      DataVector(5));
  pypp::check_with_random_values<1, initial_tags>(
      &ConstantDensityStarProxy::initial_variables, solution,
      "ConstantDensityStar",
      {"initial_conformal_factor", "initial_conformal_factor_gradient"},
      {{{-test_radius, test_radius}}}, std::make_tuple(density, radius),
      DataVector(5));
  pypp::check_with_random_values<1, source_tags>(
      &ConstantDensityStarProxy::source_variables, solution,
      "ConstantDensityStar",
      {"conformal_factor_source", "conformal_factor_gradient_source"},
      {{{-test_radius, test_radius}}}, std::make_tuple(density, radius),
      DataVector(5));

  // Test that we selected the weak-field solution of the two possible branches
  const tnsr::I<DataVector, 3> far_away_coords{{{{1e16 * radius, 0., 0.},
                                                 {0., 1e16 * radius, 0.},
                                                 {0., 0., 1e16 * radius}}}};
  const auto far_away_solution = solution.field_variables(far_away_coords);
  CHECK_ITERABLE_APPROX(
      get<Xcts::Tags::ConformalFactor<DataVector>>(far_away_solution),
      make_with_value<Scalar<DataVector>>(far_away_coords, 1.));

  Xcts::Solutions::ConstantDensityStar created_solution =
      test_creation<Xcts::Solutions::ConstantDensityStar>(options);
  CHECK(created_solution == solution);
  test_serialization(solution);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Xcts.ConstantDensityStar",
    "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Xcts"};
  test_solution(0.01, 1.,
                "  Density: 0.01\n"
                "  Radius: 1.\n");
}
