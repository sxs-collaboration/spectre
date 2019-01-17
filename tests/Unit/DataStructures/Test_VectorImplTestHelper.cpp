// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <complex>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "Utilities/ContainerHelpers.hpp"
#include "tests/Unit/DataStructures/VectorImplTestHelper.hpp"

namespace TestHelpers {
namespace VectorImpl {

SPECTRE_TEST_CASE("Unit.DataStructures.VectorImplTestHelper",
                  "[DataStructures][Unit]") {
  // testing size utility
  std::array<DataVector, 3> array_of_vectors = {
      {{{1.0, 4.0}}, {{2.0, 5.0}}, {{3.0, 6.0}}}};
  CHECK(get_size(array_of_vectors, detail::VectorOrArraySize{}) == 6);

  std::array<ComplexDataVector, 3> array_of_complex_vectors = {
      {{{std::complex<double>(1.0, 1.5), std::complex<double>(4.0, 4.5)}},
       {{std::complex<double>(2.0, 2.5), std::complex<double>(5.0, 5.5)}},
       {{std::complex<double>(3.0, 3.5), std::complex<double>(6.0, 6.5)}}}};
  CHECK(get_size(array_of_complex_vectors, detail::VectorOrArraySize{}) == 6);

  DataVector vector = {{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
  CHECK(get_size(vector, detail::VectorOrArraySize{}) == 6);

  ComplexDataVector complex_vector = {
      {std::complex<double>(1.0, 1.5), std::complex<double>(2.0, 2.5),
       std::complex<double>(3.0, 3.5), std::complex<double>(4.0, 4.5),
       std::complex<double>(5.0, 5.5), std::complex<double>(6.0, 6.5)}};
  CHECK(get_size(complex_vector, detail::VectorOrArraySize{}) == 6);

  CHECK(get_size(1.0, detail::VectorOrArraySize{}) == 1);
  CHECK(get_size(std::complex<double>(1.0, 1.5), detail::VectorOrArraySize{}) ==
        1);

  // testing indexing utility
  const std::array<DataVector, 3> constant_array_of_vectors = {
      {{{1.0, 4.0}}, {{2.0, 5.0}}, {{3.0, 6.0}}}};
  const std::array<ComplexDataVector, 3> constant_array_of_complex_vectors = {
      {{{std::complex<double>(1.0, 1.5), std::complex<double>(4.0, 4.5)}},
       {{std::complex<double>(2.0, 2.5), std::complex<double>(5.0, 5.5)}},
       {{std::complex<double>(3.0, 3.5), std::complex<double>(6.0, 6.5)}}}};

  double fundamental_to_index = 9.0;
  std::complex<double> complex_to_index = std::complex<double>(9.0, 9.5);

  CHECK(get_element(constant_array_of_vectors, 1, detail::VectorOrArrayAt{}) ==
        2.0);

  CHECK(get_element(vector, 2, detail::VectorOrArrayAt{}) == 3.0);
  get_element(vector, 2, detail::VectorOrArrayAt{}) = 10.0;
  CHECK(get_element(vector, 2, detail::VectorOrArrayAt{}) == 10.0);

  CHECK(get_element(array_of_vectors, 0, detail::VectorOrArrayAt{}) == 1.0);
  get_element(array_of_vectors, 0, detail::VectorOrArrayAt{}) = 11.0;
  CHECK(get_element(array_of_vectors, 0, detail::VectorOrArrayAt{}) == 11.0);

  CHECK(get_element(fundamental_to_index, 4, detail::VectorOrArrayAt{}) == 9.0);
  get_element(fundamental_to_index, 5, detail::VectorOrArrayAt{}) = 12.0;
  CHECK(get_element(fundamental_to_index, 0, detail::VectorOrArrayAt{}) ==
        12.0);

  CHECK(get_element(constant_array_of_complex_vectors, 1,
                    detail::VectorOrArrayAt{}) ==
        std::complex<double>(2.0, 2.5));

  CHECK(get_element(complex_vector, 2, detail::VectorOrArrayAt{}) ==
        std::complex<double>(3.0, 3.5));
  get_element(complex_vector, 2, detail::VectorOrArrayAt{}) =
      std::complex<double>(10.0, 10.5);
  CHECK(get_element(complex_vector, 2, detail::VectorOrArrayAt{}) ==
        std::complex<double>(10.0, 10.5));

  CHECK(get_element(array_of_complex_vectors, 0, detail::VectorOrArrayAt{}) ==
        std::complex<double>(1.0, 1.5));
  get_element(array_of_complex_vectors, 0, detail::VectorOrArrayAt{}) =
      std::complex<double>(11.0, 11.5);
  CHECK(get_element(array_of_complex_vectors, 0, detail::VectorOrArrayAt{}) ==
        std::complex<double>(11.0, 11.5));

  CHECK(get_element(complex_to_index, 4, detail::VectorOrArrayAt{}) ==
        std::complex<double>(9.0, 9.5));
  get_element(complex_to_index, 5, detail::VectorOrArrayAt{}) =
      std::complex<double>(12.0, 12.5);
  CHECK(get_element(complex_to_index, 0, detail::VectorOrArrayAt{}) ==
        std::complex<double>(12.0, 12.5));
}
}  // namespace VectorImpl
}  // namespace TestHelpers
