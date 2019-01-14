// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

template <size_t Dim>
struct ProductOfSinusoidsProxy : Poisson::Solutions::ProductOfSinusoids<Dim> {
  using Poisson::Solutions::ProductOfSinusoids<Dim>::ProductOfSinusoids;

  using field_tags = tmpl::list<Poisson::Field, Poisson::AuxiliaryField<Dim>>;
  using source_tags = db::wrap_tags_in<Tags::Source, field_tags>;

  tuples::tagged_tuple_from_typelist<field_tags> field_variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const noexcept {
    return Poisson::Solutions::ProductOfSinusoids<Dim>::variables(x,
                                                                  field_tags{});
  }

  tuples::tagged_tuple_from_typelist<source_tags> source_variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const noexcept {
    return Poisson::Solutions::ProductOfSinusoids<Dim>::variables(
        x, source_tags{});
  }
};

template <size_t Dim>
void test_solution(const std::array<double, Dim>& wave_numbers,
                   const std::string& options) {
  const ProductOfSinusoidsProxy<Dim> solution(wave_numbers);
  pypp::check_with_random_values<
      1, tmpl::list<Poisson::Field, Poisson::AuxiliaryField<Dim>>>(
      &ProductOfSinusoidsProxy<Dim>::field_variables, solution,
      "ProductOfSinusoids", {"field", "auxiliary_field"}, {{{0., 2. * M_PI}}},
      std::make_tuple(wave_numbers), DataVector(5));
  pypp::check_with_random_values<
      1, tmpl::list<Tags::Source<Poisson::Field>,
                    Tags::Source<Poisson::AuxiliaryField<Dim>>>>(
      &ProductOfSinusoidsProxy<Dim>::source_variables, solution,
      "ProductOfSinusoids", {"source", "auxiliary_source"}, {{{0., 2. * M_PI}}},
      std::make_tuple(wave_numbers), DataVector(5));

  Poisson::Solutions::ProductOfSinusoids<Dim> created_solution =
      test_creation<Poisson::Solutions::ProductOfSinusoids<Dim>>(
          "  WaveNumbers: " + options);
  CHECK(created_solution == solution);
  test_serialization(solution);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Poisson.ProductOfSinusoids",
    "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Poisson"};
  test_solution<1>({{0.5}}, "[0.5]");
  test_solution<2>({{0.5, 3.}}, "[0.5, 3]");
  test_solution<3>({{1., 0.5, 3.}}, "[1, 0.5, 3]");
}
