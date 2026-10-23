// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim, UpLo UpLo0, UpLo UpLo1, typename Fr,
          IndexType TypeOfIndex, typename DataType>
void test_trace_last_indices(const DataType& used_for_size) {
  using Index0 = Tensor_detail::TensorIndexType<Dim, UpLo0, Fr, TypeOfIndex>;
  using Index1 = Tensor_detail::TensorIndexType<Dim, UpLo1, Fr, TypeOfIndex>;
  pypp::check_with_random_values<1>(
      static_cast<Tensor<DataType, Symmetry<1>, index_list<Index0>> (*)(
          const Tensor<DataType, Symmetry<2, 1, 1>,
                       index_list<Index0, Index1, Index1>>&,
          const Tensor<DataType, Symmetry<1, 1>,
                       index_list<change_index_up_lo<Index1>,
                                  change_index_up_lo<Index1>>>&)>(
          &trace_last_indices<DataType, Index0, Index1>),
      "Trace", "trace_last_indices", {{{-10., 10.}}}, used_for_size);
}
template <size_t Dim, UpLo UpOrLo, typename Fr, IndexType TypeOfIndex,
          typename DataType>
void test_trace(const DataType& used_for_size) {
  using Index0 = Tensor_detail::TensorIndexType<Dim, UpOrLo, Fr, TypeOfIndex>;
  pypp::check_with_random_values<1>(
      static_cast<Scalar<DataType> (*)(
          const Tensor<DataType, Symmetry<1, 1>, index_list<Index0, Index0>>&,
          const Tensor<DataType, Symmetry<1, 1>,
                       index_list<change_index_up_lo<Index0>,
                                  change_index_up_lo<Index0>>>&)>(
          &trace<DataType, Index0>),
      "numpy", "tensordot", {{{-10., 10.}}}, used_for_size);
}

#ifdef SPECTRE_KOKKOS
// All overloads give the same results in a Kokkos kernel as on the host
void test_on_device() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  const auto rank_three = make_with_random_values<tnsr::ijj<double, 3>>(
      make_not_null(&generator), make_not_null(&distribution), 0.0);
  const auto rank_two = make_with_random_values<tnsr::ii<double, 3>>(
      make_not_null(&generator), make_not_null(&distribution), 0.0);
  const auto inverse_metric = make_with_random_values<tnsr::II<double, 3>>(
      make_not_null(&generator), make_not_null(&distribution), 0.0);
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    tnsr::i<double, 3> traced{};
    trace_last_indices(make_not_null(&traced), rank_three, inverse_metric);
    const auto returned_traced = trace_last_indices(rank_three, inverse_metric);
    Scalar<double> scalar_trace{};
    ::trace(make_not_null(&scalar_trace), rank_two, inverse_metric);
    return std::array<double, 8>{{get<0>(traced), get<1>(traced),
                                  get<2>(traced), get<0>(returned_traced),
                                  get<1>(returned_traced),
                                  get<2>(returned_traced), get(scalar_trace),
                                  get(::trace(rank_two, inverse_metric))}};
  });
}
#endif  // SPECTRE_KOKKOS
}  // namespace

SPECTRE_TEST_CASE("Unit.Tensor.EagerMath.Trace", "[DataStructures][Unit]") {
  const pypp::SetupLocalPythonEnvironment local_python_env(
      "DataStructures/Tensor/EagerMath/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_trace_last_indices, (1, 2, 3),
                                    (UpLo::Lo, UpLo::Up), (UpLo::Lo, UpLo::Up),
                                    (Frame::Grid, Frame::Inertial),
                                    (IndexType::Spatial, IndexType::Spacetime));

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_trace, (1, 2, 3), (UpLo::Lo, UpLo::Up),
                                    (Frame::Grid, Frame::Inertial),
                                    (IndexType::Spatial, IndexType::Spacetime));
#ifdef SPECTRE_KOKKOS
  test_on_device();
#endif  // SPECTRE_KOKKOS
}
