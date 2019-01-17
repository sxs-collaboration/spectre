// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <complex>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Variables
// IWYU pragma: no_forward_declare Tensor

namespace {
template <size_t Dim>
struct Var1 {
  using type = tnsr::i<DataVector, Dim, Frame::Grid>;
};

struct Var2 {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
using two_vars = tmpl::list<Var1<Dim>, Var2>;

template <size_t Dim>
using one_var = tmpl::list<Var1<Dim>>;

template <typename R, typename T, typename ValueType>
void check_make_with_value(const R& expected, const T& input,
                           const ValueType value) {
  const auto computed = make_with_value<R>(input, value);
  CHECK(expected == computed);
}

void test_make_tagged_tuple() {
  for (size_t n_pts = 1; n_pts < 4; ++n_pts) {
    check_make_with_value(
        tuples::TaggedTuple<Var2>(Scalar<DataVector>(n_pts, -5.7)),
        DataVector(n_pts, 0.0), -5.7);
    check_make_with_value(
        tuples::TaggedTuple<Var2>(Scalar<DataVector>(n_pts, -5.7)),
        tnsr::ab<DataVector, 2, Frame::Inertial>(n_pts, 0.0), -5.7);

    check_make_with_value(tuples::TaggedTuple<Var1<3>, Var2>(
                              tnsr::i<DataVector, 3, Frame::Grid>(n_pts, 3.8),
                              Scalar<DataVector>(n_pts, 3.8)),
                          DataVector(n_pts, 0.0), 3.8);

    check_make_with_value(tuples::TaggedTuple<Var1<3>, Var2>(
                              tnsr::i<DataVector, 3, Frame::Grid>(n_pts, 3.8),
                              Scalar<DataVector>(n_pts, 3.8)),
                          tnsr::ab<DataVector, 2, Frame::Inertial>(n_pts, 0.0),
                          3.8);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.MakeWithValue",
                  "[DataStructures][Unit]") {
  check_make_with_value(8.3, 1.3, 8.3);
  check_make_with_value(std::complex<double>(8.3, 2.5), 1.3,
                        std::complex<double>(8.3, 2.5));
  check_make_with_value(8.3, tnsr::i<double, 3>{{{1.3, 8.3, 3.4}}}, 8.3);
  check_make_with_value(std::complex<double>(8.3, 2.5),
                        tnsr::i<double, 3>{{{1.3, 8.3, 3.4}}},
                        std::complex<double>(8.3, 2.5));
  check_make_with_value(8.3, DataVector{8, 2.3}, 8.3);
  check_make_with_value(std::complex<double>(8.3, 2.5), DataVector{8, 2.3},
                        std::complex<double>(8.3, 2.5));
  check_make_with_value(
      8.3,
      tnsr::i<DataVector, 3>{
          {{DataVector{8, 2.3}, DataVector{8, 9.8}, DataVector{8, -1.2}}}},
      8.3);
  check_make_with_value(
      std::complex<double>(8.3, 2.5),
      tnsr::i<DataVector, 3>{
          {{DataVector{8, 2.3}, DataVector{8, 9.8}, DataVector{8, -1.2}}}},
      std::complex<double>(8.3, 2.5));

  check_make_with_value(Scalar<double>(8.3), 1.3, 8.3);
  check_make_with_value(tnsr::I<double, 3, Frame::Grid>(8.3), 1.3, 8.3);
  check_make_with_value(8.3, tnsr::I<double, 3, Frame::Grid>(1.3), 8.3);
  check_make_with_value(std::complex<double>(8.3, 2.5),
                        tnsr::I<double, 3, Frame::Grid>(1.3),
                        std::complex<double>(8.3, 2.5));
  check_make_with_value(tnsr::Ij<double, 3, Frame::Grid>(8.3),
                        tnsr::aB<double, 1, Frame::Inertial>(1.3), 8.3);
  check_make_with_value(make_array<4>(8.3), 1.3, 8.3);

  for (size_t n_pts = 1; n_pts < 4; ++n_pts) {
    // create DataVector from DataVector
    check_make_with_value(make_array<3>(DataVector(n_pts, 8.3)),
                          DataVector(n_pts, 4.5), 8.3);
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
  test_make_tagged_tuple();
}
