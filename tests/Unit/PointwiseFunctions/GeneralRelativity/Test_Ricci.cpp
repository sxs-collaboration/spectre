// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Unit/PointwiseFunctions/GeneralRelativity/GrTestHelpers.hpp"

namespace {
template <typename DataType>
void test_1d_spatial_ricci(const DataType& used_for_size) {
  constexpr size_t spatial_dim = 1;
  const auto ricci = gr::ricci_tensor(
      make_spatial_christoffel_second_kind<spatial_dim>(
          used_for_size),
      make_deriv_spatial_christoffel_second_kind<spatial_dim>(
          used_for_size));
  CHECK_ITERABLE_APPROX(ricci.get(0, 0),
                        make_with_value<DataType>(used_for_size, 0.));
}

template <typename DataType>
void test_2d_spatial_ricci(const DataType& used_for_size) {
  constexpr size_t spatial_dim = 2;
  const auto ricci = gr::ricci_tensor(
      make_spatial_christoffel_second_kind<spatial_dim>(
          used_for_size),
      make_deriv_spatial_christoffel_second_kind<spatial_dim>(
          used_for_size));
  CHECK_ITERABLE_APPROX(ricci.get(0, 0),
                        make_with_value<DataType>(used_for_size, -9.));
  CHECK_ITERABLE_APPROX(ricci.get(0, 1),
                        make_with_value<DataType>(used_for_size, 5.5));
  CHECK_ITERABLE_APPROX(ricci.get(1, 1),
                        make_with_value<DataType>(used_for_size, 20.));
}

template <typename DataType>
void test_3d_spatial_ricci(const DataType& used_for_size) {
  constexpr size_t spatial_dim = 3;
  const auto ricci = gr::ricci_tensor(
      make_spatial_christoffel_second_kind<spatial_dim>(
          used_for_size),
      make_deriv_spatial_christoffel_second_kind<spatial_dim>(
          used_for_size));

  CHECK_ITERABLE_APPROX(ricci.get(0, 0),
                        make_with_value<DataType>(used_for_size, -65.));
  CHECK_ITERABLE_APPROX(ricci.get(0, 1),
                        make_with_value<DataType>(used_for_size, 2.5));
  CHECK_ITERABLE_APPROX(ricci.get(0, 2),
                        make_with_value<DataType>(used_for_size, 70.));
  CHECK_ITERABLE_APPROX(ricci.get(1, 1),
                        make_with_value<DataType>(used_for_size, 46.));
  CHECK_ITERABLE_APPROX(ricci.get(1, 2),
                        make_with_value<DataType>(used_for_size, 89.5));
  CHECK_ITERABLE_APPROX(ricci.get(2, 2),
                        make_with_value<DataType>(used_for_size, 109.));
}

template <typename DataType>
void test_1d_spacetime_ricci(const DataType& used_for_size) {
  constexpr size_t spatial_dim = 1;
  const auto ricci = gr::ricci_tensor(
      make_spacetime_christoffel_second_kind<spatial_dim>(
          used_for_size),
      make_deriv_spacetime_christoffel_second_kind<spatial_dim>(
          used_for_size));
  CHECK_ITERABLE_APPROX(ricci.get(0, 0),
                        make_with_value<DataType>(used_for_size, -46.));
  CHECK_ITERABLE_APPROX(ricci.get(0, 1),
                        make_with_value<DataType>(used_for_size, 28.));
  CHECK_ITERABLE_APPROX(ricci.get(1, 1),
                        make_with_value<DataType>(used_for_size, -16.));
}

template <typename DataType>
void test_2d_spacetime_ricci(const DataType& used_for_size) {
  constexpr size_t spatial_dim = 2;
  const auto ricci = gr::ricci_tensor(
      make_spacetime_christoffel_second_kind<spatial_dim>(
          used_for_size),
      make_deriv_spacetime_christoffel_second_kind<spatial_dim>(
          used_for_size));
  CHECK_ITERABLE_APPROX(ricci.get(0, 0),
                        make_with_value<DataType>(used_for_size, -406.));
  CHECK_ITERABLE_APPROX(ricci.get(0, 1),
                        make_with_value<DataType>(used_for_size, -32.));
  CHECK_ITERABLE_APPROX(ricci.get(0, 2),
                        make_with_value<DataType>(used_for_size, 217.));
  CHECK_ITERABLE_APPROX(ricci.get(1, 1),
                        make_with_value<DataType>(used_for_size, -178.));
  CHECK_ITERABLE_APPROX(ricci.get(1, 2),
                        make_with_value<DataType>(used_for_size, 143.5));
  CHECK_ITERABLE_APPROX(ricci.get(2, 2),
                        make_with_value<DataType>(used_for_size, -201.));
}

template <typename DataType>
void test_3d_spacetime_ricci(const DataType& used_for_size) {
  constexpr size_t spatial_dim = 3;
  const auto ricci = gr::ricci_tensor(
      make_spacetime_christoffel_second_kind<spatial_dim>(
          used_for_size),
      make_deriv_spacetime_christoffel_second_kind<spatial_dim>(
          used_for_size));
  CHECK_ITERABLE_APPROX(ricci.get(0, 0),
                        make_with_value<DataType>(used_for_size, -1928.));
  CHECK_ITERABLE_APPROX(ricci.get(0, 1),
                        make_with_value<DataType>(used_for_size, -700.));
  CHECK_ITERABLE_APPROX(ricci.get(0, 2),
                        make_with_value<DataType>(used_for_size, 283.));
  CHECK_ITERABLE_APPROX(ricci.get(0, 3),
                        make_with_value<DataType>(used_for_size, 940.));
  CHECK_ITERABLE_APPROX(ricci.get(1, 1),
                        make_with_value<DataType>(used_for_size, -1300.));
  CHECK_ITERABLE_APPROX(ricci.get(1, 2),
                        make_with_value<DataType>(used_for_size, 82.5));
  CHECK_ITERABLE_APPROX(ricci.get(1, 3),
                        make_with_value<DataType>(used_for_size, 968.));
  CHECK_ITERABLE_APPROX(ricci.get(2, 2),
                        make_with_value<DataType>(used_for_size, -629.));
  CHECK_ITERABLE_APPROX(ricci.get(2, 3),
                        make_with_value<DataType>(used_for_size, 335.5));
  CHECK_ITERABLE_APPROX(ricci.get(3, 3),
                        make_with_value<DataType>(used_for_size, -1176.));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.Ricci.",
                  "[PointwiseFunctions][Unit]") {
  const double d(std::numeric_limits<double>::signaling_NaN());
  test_1d_spatial_ricci(d);
  test_2d_spatial_ricci(d);
  test_3d_spatial_ricci(d);
  test_1d_spacetime_ricci(d);
  test_2d_spacetime_ricci(d);
  test_3d_spacetime_ricci(d);

  const DataVector dv(5);
  test_1d_spatial_ricci(dv);
  test_2d_spatial_ricci(dv);
  test_3d_spatial_ricci(dv);
  test_1d_spacetime_ricci(dv);
  test_2d_spacetime_ricci(dv);
  test_3d_spacetime_ricci(dv);
}
