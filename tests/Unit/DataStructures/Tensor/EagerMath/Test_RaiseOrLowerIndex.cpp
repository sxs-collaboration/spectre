// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim, UpLo UpOrLo, IndexType Index, typename DataType>
void test_raise_or_lower_first_index(const DataType& used_for_size) {
  using Index0 =
      Tensor_detail::TensorIndexType<Dim, UpOrLo, Frame::Inertial, Index>;
  using Index1 =
      Tensor_detail::TensorIndexType<Dim, UpLo::Lo, Frame::Inertial, Index>;
  pypp::check_with_random_values<1>(
      static_cast<
          Tensor<DataType, Symmetry<2, 1, 1>,
                 index_list<change_index_up_lo<Index0>, Index1, Index1>> (*)(
              const Tensor<DataType, Symmetry<2, 1, 1>,
                           index_list<Index0, Index1, Index1>>&,
              const Tensor<DataType, Symmetry<1, 1>,
                           index_list<change_index_up_lo<Index0>,
                                      change_index_up_lo<Index0>>>&)>(
          &raise_or_lower_first_index<DataType, Index0, Index1>),
      "RaiseOrLowerIndex", "raise_or_lower_first_index", {{{-10., 10.}}},
      used_for_size);
}

template <size_t Dim, UpLo UpOrLo, IndexType Index, typename DataType>
void test_raise_or_lower(const DataType& used_for_size) {
  using Index0 =
      Tensor_detail::TensorIndexType<Dim, UpOrLo, Frame::Inertial, Index>;
  pypp::check_with_random_values<1>(
      static_cast<Tensor<DataType, Symmetry<1>,
                         index_list<change_index_up_lo<Index0>>> (*)(
          const Tensor<DataType, Symmetry<1>, index_list<Index0>>&,
          const Tensor<DataType, Symmetry<1, 1>,
                       index_list<change_index_up_lo<Index0>,
                                  change_index_up_lo<Index0>>>&)>(
          &raise_or_lower_index<DataType, DataType, Index0>),
      "numpy", "matmul", {{{-10., 10.}}}, used_for_size);
}

#ifdef SPECTRE_KOKKOS
// All overloads give the same results in a Kokkos kernel as on the host
void test_on_device() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  const auto one_form = make_with_random_values<tnsr::i<double, 3>>(
      make_not_null(&generator), make_not_null(&distribution), 0.0);
  const auto tensor = make_with_random_values<tnsr::ijj<double, 3>>(
      make_not_null(&generator), make_not_null(&distribution), 0.0);
  const auto inverse_metric = make_with_random_values<tnsr::II<double, 3>>(
      make_not_null(&generator), make_not_null(&distribution), 0.0);
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    tnsr::I<double, 3> raised{};
    raise_or_lower_index(make_not_null(&raised), one_form, inverse_metric);
    const auto returned_raised = raise_or_lower_index(one_form, inverse_metric);
    tnsr::Ijj<double, 3> raised_first{};
    raise_or_lower_first_index(make_not_null(&raised_first), tensor,
                               inverse_metric);
    const auto returned_raised_first =
        raise_or_lower_first_index(tensor, inverse_metric);
    // 3 components of the vector and 18 of the rank-3 tensor, from both
    // overloads
    Kokkos::Array<double, 42> result{};
    size_t n = 0;
    for (size_t i = 0; i < 3; ++i) {
      result[n++] = raised.get(i);
      result[n++] = returned_raised.get(i);
      for (size_t j = 0; j < 3; ++j) {
        for (size_t k = j; k < 3; ++k) {
          result[n++] = raised_first.get(i, j, k);
          result[n++] = returned_raised_first.get(i, j, k);
        }
      }
    }
    return result;
  });
}
#endif  // SPECTRE_KOKKOS
}  // namespace

SPECTRE_TEST_CASE("Unit.Tensor.EagerMath.RaiseOrLowerIndex",
                  "[DataStructures][Unit]") {
  const pypp::SetupLocalPythonEnvironment local_python_env(
      "DataStructures/Tensor/EagerMath/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_raise_or_lower_first_index, (1, 2, 3),
                                    (UpLo::Lo, UpLo::Up),
                                    (IndexType::Spatial, IndexType::Spacetime));

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_raise_or_lower, (1, 2, 3),
                                    (UpLo::Lo, UpLo::Up),
                                    (IndexType::Spatial, IndexType::Spacetime));
#ifdef SPECTRE_KOKKOS
  test_on_device();
#endif  // SPECTRE_KOKKOS
}
