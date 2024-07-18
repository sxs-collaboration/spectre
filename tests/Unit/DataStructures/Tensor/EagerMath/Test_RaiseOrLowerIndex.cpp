// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Utilities/TMPL.hpp"

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
  pypp::check_with_random_values<1>(f, "RaiseOrLowerIndex",
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
}  // namespace

SPECTRE_TEST_CASE("Unit.Tensor.EagerMath.RaiseOrLowerIndex",
                  "[DataStructures][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "DataStructures/Tensor/EagerMath/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_raise_or_lower_first_index, (1, 2, 3),
                                    (UpLo::Lo, UpLo::Up),
                                    (IndexType::Spatial, IndexType::Spacetime));

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_raise_or_lower, (1, 2, 3),
                                    (UpLo::Lo, UpLo::Up),
                                    (IndexType::Spatial, IndexType::Spacetime));
}
