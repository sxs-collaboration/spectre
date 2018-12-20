// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>

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

  DataVector vector = {{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
  CHECK(get_size(vector, detail::VectorOrArraySize{}) == 6);

  CHECK(get_size(1.0, detail::VectorOrArraySize{}) == 1);

  // testing indexing utility
  const std::array<DataVector, 3> constant_array_of_vectors = {
      {{{1.0, 4.0}}, {{2.0, 5.0}}, {{3.0, 6.0}}}};
  double fundamental_to_index = 9.0;

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
}
}  // namespace VectorImpl
}  // namespace TestHelpers
