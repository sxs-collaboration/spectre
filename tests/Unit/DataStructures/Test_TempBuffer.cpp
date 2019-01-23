// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
template <typename DataType>
void test_temp_buffer(const DataType& x) {
  TempBuffer<tmpl::list<::Tags::TempI<0, 3, Frame::Inertial, DataType>,
                        ::Tags::TempScalar<1, DataType>>>
      buffer(get_size(x));

  auto& vec = get<::Tags::TempI<0, 3, Frame::Inertial, DataType>>(buffer);
  auto& scalar = get<::Tags::TempScalar<1, DataType>>(buffer);

  // Do a few operations on the vector and scalar.
  get<0>(vec) = x + 1.0;
  get<1>(vec) = square(x);
  get<2>(vec) = 6.0 * x;

  scalar = magnitude(vec);

  const auto expected_scalar = make_with_value<Scalar<DataType>>(x, 13.0);

  // Get a separate handle `scalar2` to make sure that the value in
  // the buffer has changed.
  auto& scalar2 = get<::Tags::TempScalar<1, DataType>>(buffer);
  CHECK_ITERABLE_APPROX(scalar2, expected_scalar);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.TempBuffer", "[DataStructures][Unit]") {
  test_temp_buffer(2.0);
  test_temp_buffer(DataVector(5, 2.0));
}
