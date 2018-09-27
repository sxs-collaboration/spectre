// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

// IWYU pragma: no_include <boost/preprocessor/arithmetic/dec.hpp>
// IWYU pragma: no_include <boost/preprocessor/repetition/enum.hpp>
// IWYU pragma: no_include <boost/preprocessor/tuple/reverse.hpp>

namespace {
template <size_t Dim, UpLo UpOrLo, IndexType Index, typename DataType>
void test_raise_or_lower_first_index(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      &raise_or_lower_first_index<
          DataType,
          Tensor_detail::TensorIndexType<Dim, UpOrLo, Frame::Inertial, Index>,
          Tensor_detail::TensorIndexType<Dim, UpLo::Lo, Frame::Inertial,
                                         Index>>,
      "TestFunctions", "raise_or_lower_first_index", {{{-10., 10.}}},
      used_for_size);
}

template <size_t Dim, UpLo UpOrLo, IndexType Index, typename DataType>
void test_raise_or_lower(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      &raise_or_lower_index<DataType, Tensor_detail::TensorIndexType<
                                          Dim, UpOrLo, Frame::Inertial, Index>>,
      "numpy", "matmul", {{{-10., 10.}}}, used_for_size);
}
template <size_t Dim, UpLo UpLo0, UpLo UpLo1, typename Fr,
          IndexType TypeOfIndex, typename DataType>
void test_trace_last_indices(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      &trace_last_indices<
          DataType, Tensor_detail::TensorIndexType<Dim, UpLo0, Fr, TypeOfIndex>,
          Tensor_detail::TensorIndexType<Dim, UpLo1, Fr, TypeOfIndex>>,
      "TestFunctions", "trace_last_indices", {{{-10., 10.}}}, used_for_size);
}
template <size_t Dim, UpLo UpOrLo, typename Fr, IndexType TypeOfIndex,
          typename DataType>
void test_trace(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      &trace<DataType,
             Tensor_detail::TensorIndexType<Dim, UpOrLo, Fr, TypeOfIndex>>,
      "numpy", "tensordot", {{{-10., 10.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.IndexManipulation",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_raise_or_lower_first_index, (1, 2, 3),
                                    (UpLo::Lo, UpLo::Up),
                                    (IndexType::Spatial, IndexType::Spacetime));

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_raise_or_lower, (1, 2, 3),
                                    (UpLo::Lo, UpLo::Up),
                                    (IndexType::Spatial, IndexType::Spacetime));

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_trace_last_indices, (1, 2, 3),
                                    (UpLo::Lo, UpLo::Up), (UpLo::Lo, UpLo::Up),
                                    (Frame::Grid, Frame::Inertial),
                                    (IndexType::Spatial, IndexType::Spacetime));

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_trace, (1, 2, 3), (UpLo::Lo, UpLo::Up),
                                    (Frame::Grid, Frame::Inertial),
                                    (IndexType::Spatial, IndexType::Spacetime));
}
