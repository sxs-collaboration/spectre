// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/OuterProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

template <typename DataType>
void test(const DataType& used_for_size) {
  // Explicit types in return values to check return types (symmetry etc)
  {
    const tnsr::iJ<DataType, 1> result =
        outer_product(make_with_value<tnsr::i<DataType, 1>>(used_for_size, 2.),
                      make_with_value<tnsr::I<DataType, 1>>(used_for_size, 4.));
    CHECK_ITERABLE_APPROX((get<0, 0>(result)),
                          make_with_value<DataType>(used_for_size, 8.));
  }
  {
    const tnsr::ijj<DataType, 1> result = outer_product(
        make_with_value<tnsr::i<DataType, 1>>(used_for_size, 2.),
        make_with_value<tnsr::ii<DataType, 1>>(used_for_size, 4.));
    CHECK_ITERABLE_APPROX((get<0, 0, 0>(result)),
                          make_with_value<DataType>(used_for_size, 8.));
  }
  {
    const tnsr::Ijk<DataType, 1> result = outer_product(
        make_with_value<tnsr::I<DataType, 1>>(used_for_size, 2.),
        make_with_value<tnsr::ij<DataType, 1>>(used_for_size, 4.));
    CHECK_ITERABLE_APPROX((get<0, 0, 0>(result)),
                          make_with_value<DataType>(used_for_size, 8.));
  }
  {
    tnsr::i<DataType, 2> lhs{};
    get<0>(lhs) = make_with_value<DataType>(used_for_size, 1.);
    get<1>(lhs) = make_with_value<DataType>(used_for_size, 2.);
    tnsr::Ij<DataType, 2> rhs{};
    get<0, 0>(rhs) = make_with_value<DataType>(used_for_size, 1.);
    get<0, 1>(rhs) = make_with_value<DataType>(used_for_size, 2.);
    get<1, 0>(rhs) = make_with_value<DataType>(used_for_size, 3.);
    get<1, 1>(rhs) = make_with_value<DataType>(used_for_size, 4.);
    const tnsr::iJk<DataType, 2> result = outer_product(lhs, rhs);
    CHECK_ITERABLE_APPROX((get<0, 0, 0>(result)),
                          make_with_value<DataType>(used_for_size, 1.));
    CHECK_ITERABLE_APPROX((get<0, 0, 1>(result)),
                          make_with_value<DataType>(used_for_size, 2.));
    CHECK_ITERABLE_APPROX((get<0, 1, 0>(result)),
                          make_with_value<DataType>(used_for_size, 3.));
    CHECK_ITERABLE_APPROX((get<0, 1, 1>(result)),
                          make_with_value<DataType>(used_for_size, 4.));
    CHECK_ITERABLE_APPROX((get<1, 0, 0>(result)),
                          make_with_value<DataType>(used_for_size, 2.));
    CHECK_ITERABLE_APPROX((get<1, 0, 1>(result)),
                          make_with_value<DataType>(used_for_size, 4.));
    CHECK_ITERABLE_APPROX((get<1, 1, 0>(result)),
                          make_with_value<DataType>(used_for_size, 6.));
    CHECK_ITERABLE_APPROX((get<1, 1, 1>(result)),
                          make_with_value<DataType>(used_for_size, 8.));
  }
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.EagerMath.OuterProduct",
                  "[DataStructures][Unit]") {
  test(DataVector(3));
  test(3.);
}
