// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

// IWYU pragma: no_include <boost/preprocessor/arithmetic/dec.hpp>
// IWYU pragma: no_include <boost/preprocessor/repetition/enum.hpp>
// IWYU pragma: no_include <boost/preprocessor/tuple/reverse.hpp>

// IWYU pragma: no_forward_declare Tensor

namespace {
template <size_t Dim, UpLo UpOrLo, IndexType Index, typename DataType>
void test_raise_or_lower_first_index(const DataType& used_for_size) {
  using Index0 =
      Tensor_detail::TensorIndexType<Dim, UpOrLo, Frame::Inertial, Index>;
  using Index1 =
      Tensor_detail::TensorIndexType<Dim, UpLo::Lo, Frame::Inertial, Index>;
  Tensor<DataType, Symmetry<2, 1, 1>,
         index_list<change_index_up_lo<Index0>, Index1, Index1>> (*f)(
      const Tensor<DataType, Symmetry<2, 1, 1>,
                   index_list<Index0, Index1, Index1>>&,
      const Tensor<DataType, Symmetry<1, 1>,
                   index_list<change_index_up_lo<Index0>,
                              change_index_up_lo<Index0>>>&) =
      &raise_or_lower_first_index<DataType, Index0, Index1>;
  pypp::check_with_random_values<1>(f, "IndexManipulation",
                                    "raise_or_lower_first_index",
                                    {{{-10., 10.}}}, used_for_size);
}

template <size_t Dim, UpLo UpOrLo, IndexType Index, typename DataType>
void test_raise_or_lower(const DataType& used_for_size) {
  using Index0 =
      Tensor_detail::TensorIndexType<Dim, UpOrLo, Frame::Inertial, Index>;
  Tensor<DataType, Symmetry<1>, index_list<change_index_up_lo<Index0>>> (*f)(
      const Tensor<DataType, Symmetry<1>, index_list<Index0>>&,
      const Tensor<DataType, Symmetry<1, 1>,
                   index_list<change_index_up_lo<Index0>,
                              change_index_up_lo<Index0>>>&) =
      &raise_or_lower_index<DataType, Index0>;
  pypp::check_with_random_values<1>(f, "numpy", "matmul", {{{-10., 10.}}},
                                    used_for_size);
}
template <size_t Dim, UpLo UpLo0, UpLo UpLo1, typename Fr,
          IndexType TypeOfIndex, typename DataType>
void test_trace_last_indices(const DataType& used_for_size) {
  using Index0 = Tensor_detail::TensorIndexType<Dim, UpLo0, Fr, TypeOfIndex>;
  using Index1 = Tensor_detail::TensorIndexType<Dim, UpLo1, Fr, TypeOfIndex>;
  Tensor<DataType, Symmetry<1>, index_list<Index0>> (*f)(
      const Tensor<DataType, Symmetry<2, 1, 1>,
                   index_list<Index0, Index1, Index1>>&,
      const Tensor<DataType, Symmetry<1, 1>,
                   index_list<change_index_up_lo<Index1>,
                              change_index_up_lo<Index1>>>&) =
      &trace_last_indices<DataType, Index0, Index1>;
  pypp::check_with_random_values<1>(f, "IndexManipulation",
                                    "trace_last_indices", {{{-10., 10.}}},
                                    used_for_size);
}
template <size_t Dim, UpLo UpOrLo, typename Fr, IndexType TypeOfIndex,
          typename DataType>
void test_trace(const DataType& used_for_size) {
  using Index0 = Tensor_detail::TensorIndexType<Dim, UpOrLo, Fr, TypeOfIndex>;
  Scalar<DataType> (*f)(
      const Tensor<DataType, Symmetry<1, 1>, index_list<Index0, Index0>>&,
      const Tensor<DataType, Symmetry<1, 1>,
                   index_list<change_index_up_lo<Index0>,
                              change_index_up_lo<Index0>>>&) =
      &trace<DataType, Index0>;
  pypp::check_with_random_values<1>(f, "numpy", "tensordot", {{{-10., 10.}}},
                                    used_for_size);
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
