// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/MakeWithValue.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
template <size_t Dim>
struct Var1 {
  using type = tnsr::i<DataVector, Dim, Frame::Grid>;
};

struct Var2 {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
using two_vars = typelist<Var1<Dim>, Var2>;

template <size_t Dim>
using one_var = typelist<Var1<Dim>>;

template <typename R, typename T>
void check_make_with_value(const R& expected, const T& input,
                           const double value) {
  const auto computed = make_with_value<R>(input, value);
  CHECK(expected == computed);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.MakeWithValue",
                  "[DataStructures][Unit]") {
  check_make_with_value(8.3, 1.3, 8.3);
  check_make_with_value(Scalar<double>(8.3), 1.3, 8.3);
  check_make_with_value(tnsr::I<double, 3, Frame::Grid>(8.3), 1.3, 8.3);
  check_make_with_value(8.3, tnsr::I<double, 3, Frame::Grid>(1.3), 8.3);
  check_make_with_value(tnsr::Ij<double, 3, Frame::Grid>(8.3),
                        tnsr::aB<double, 1, Frame::Inertial>(1.3), 8.3);

  for (size_t n_pts = 1; n_pts < 4; ++n_pts) {
    // create DataVector from DataVector
    check_make_with_value(DataVector(n_pts, -2.3), DataVector(n_pts, 4.5),
                          -2.3);
    check_make_with_value(Scalar<DataVector>(n_pts, -2.3),
                          DataVector(n_pts, 4.5), -2.3);
    check_make_with_value(tnsr::ab<DataVector, 2, Frame::Inertial>(n_pts, -2.3),
                          DataVector(n_pts, 4.5), -2.3);
    check_make_with_value(DataVector(n_pts, -2.3),
                          Scalar<DataVector>(n_pts, 4.5), -2.3);
    check_make_with_value(DataVector(n_pts, -2.3),
                          tnsr::ab<DataVector, 2, Frame::Inertial>(n_pts, 4.5),
                          -2.3);
    check_make_with_value(
        tnsr::ijj<DataVector, 2, Frame::Inertial>(n_pts, -2.3),
        tnsr::ab<DataVector, 3, Frame::Grid>(n_pts, 4.5), -2.3);
    check_make_with_value(
        tnsr::ijj<DataVector, 2, Frame::Inertial>(n_pts, -2.3),
        Scalar<DataVector>(n_pts, 4.5), -2.3);
    check_make_with_value(Scalar<DataVector>(n_pts, -2.3),
                          tnsr::abc<DataVector, 3, Frame::Inertial>(n_pts, 4.5),
                          -2.3);
    check_make_with_value(Variables<two_vars<3>>(n_pts, -2.3),
                          DataVector(n_pts, 4.5), -2.3);
    check_make_with_value(Variables<one_var<3>>(n_pts, -2.3),
                          DataVector(n_pts, 4.5), -2.3);
    check_make_with_value(Variables<two_vars<3>>(n_pts, -2.3),
                          tnsr::ab<DataVector, 3, Frame::Grid>(n_pts, 4.5),
                          -2.3);
    check_make_with_value(Variables<one_var<3>>(n_pts, -2.3),
                          Scalar<DataVector>(n_pts, 4.5), -2.3);
    check_make_with_value(Variables<one_var<3>>(n_pts, -2.3),
                          Variables<two_vars<3>>(n_pts, 4.5), -2.3);
    check_make_with_value(Variables<two_vars<3>>(n_pts, -2.3),
                          Variables<one_var<3>>(n_pts, 4.5), -2.3);
  }
}
