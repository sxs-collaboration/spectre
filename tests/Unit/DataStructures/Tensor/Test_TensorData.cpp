// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep

namespace {
template <typename DataType>
void test() {
  TensorComponent tc0("T_x", DataType{8.9, 7.6, -3.4, 9.0});
  TensorComponent tc1("T_y", DataType{8.9, 7.6, -3.4, 9.0});
  TensorComponent tc2("T_x", DataType{8.9, 7.6, -3.4, 9.1});
  CHECK(get_output(tc0) == "(T_x, (8.9,7.6,-3.4,9))");
  CHECK(tc0 == tc0);
  CHECK(tc0 != tc1);
  CHECK(tc0 != tc2);
  CHECK(tc1 != tc2);
  test_serialization(tc0);

  ExtentsAndTensorVolumeData etvd0({2, 2}, {tc0, tc1, tc2});
  const auto after = serialize_and_deserialize(etvd0);
  CHECK(after.extents == etvd0.extents);
  CHECK(after.tensor_components == etvd0.tensor_components);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.TensorData", "[Unit]") {
  test<DataVector>();
  test<std::vector<float>>();
}
