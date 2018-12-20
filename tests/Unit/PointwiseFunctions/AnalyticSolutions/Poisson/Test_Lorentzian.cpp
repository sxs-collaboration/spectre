// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Lorentzian.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

template <size_t Dim>
struct LorentzianProxy : Poisson::Solutions::Lorentzian<Dim> {
  using Poisson::Solutions::Lorentzian<Dim>::Lorentzian;

  using field_tags = tmpl::list<Poisson::Field>;
  using source_tags = tmpl::list<Tags::Source<Poisson::Field>,
                                 Tags::Source<Poisson::AuxiliaryField<Dim>>>;

  tuples::tagged_tuple_from_typelist<field_tags> field_variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const noexcept {
    return Poisson::Solutions::Lorentzian<Dim>::variables(x, field_tags{});
  }

  tuples::tagged_tuple_from_typelist<source_tags> source_variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const noexcept {
    return Poisson::Solutions::Lorentzian<Dim>::variables(x, source_tags{});
  }
};

template <size_t Dim>
void test_solution() {
  const LorentzianProxy<Dim> solution{};
  pypp::check_with_random_values<1, tmpl::list<Poisson::Field>>(
      &LorentzianProxy<Dim>::field_variables, solution, "Lorentzian", {"field"},
      {{{-5., 5.}}}, std::make_tuple(), DataVector(5));
  pypp::check_with_random_values<
      1, tmpl::list<Tags::Source<Poisson::Field>,
                    Tags::Source<Poisson::AuxiliaryField<Dim>>>>(
      &LorentzianProxy<Dim>::source_variables, solution, "Lorentzian",
      {"source", "auxiliary_source"}, {{{-5., 5.}}}, std::make_tuple(),
      DataVector(5));

  const Poisson::Solutions::Lorentzian<Dim> check_solution{};
  const Poisson::Solutions::Lorentzian<Dim> created_solution =
      test_creation<Poisson::Solutions::Lorentzian<Dim>>("  ");
  CHECK(created_solution == check_solution);
  test_serialization(check_solution);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Poisson.Lorentzian",
    "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Poisson"};
  // 1D and 2D solutions are not implemented yet.
  test_solution<3>();
}
