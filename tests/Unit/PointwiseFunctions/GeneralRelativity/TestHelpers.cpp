// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace TestHelpers {
namespace gr {
template <typename DataType>
Scalar<DataType> random_lapse(const gsl::not_null<std::mt19937*> generator,
                              const DataType& used_for_size) noexcept {
  std::uniform_real_distribution<> distribution(0.0, 3.0);
  return make_with_random_values<Scalar<DataType>>(
      generator, make_not_null(&distribution), used_for_size);
}

template <size_t Dim, typename DataType>
tnsr::I<DataType, Dim> random_shift(
    const gsl::not_null<std::mt19937*> generator,
    const DataType& used_for_size) noexcept {
  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  return make_with_random_values<tnsr::I<DataType, Dim>>(
      generator, make_not_null(&distribution), used_for_size);
}

template <size_t Dim, typename DataType>
tnsr::ii<DataType, Dim> random_spatial_metric(
    const gsl::not_null<std::mt19937*> generator,
    const DataType& used_for_size) noexcept {
  std::uniform_real_distribution<> distribution(-0.05, 0.05);
  auto spatial_metric = make_with_random_values<tnsr::ii<DataType, Dim>>(
      generator, make_not_null(&distribution), used_for_size);
  for (size_t d = 0; d < Dim; ++d) {
    spatial_metric.get(d, d) += 1.0;
  }
  return spatial_metric;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                \
  template Scalar<DTYPE(data)> random_lapse(        \
      const gsl::not_null<std::mt19937*> generator, \
      const DTYPE(data) & used_for_size) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALARS, (double, DataVector))

#define INSTANTIATE_TENSORS(_, data)                               \
  template tnsr::I<DTYPE(data), DIM(data)> random_shift(           \
      const gsl::not_null<std::mt19937*> generator,                \
      const DTYPE(data) & used_for_size) noexcept;                 \
  template tnsr::ii<DTYPE(data), DIM(data)> random_spatial_metric( \
      const gsl::not_null<std::mt19937*> generator,                \
      const DTYPE(data) & used_for_size) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_TENSORS, (double, DataVector), (1, 2, 3))

#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_TENSORS
#undef DIM
#undef DTYPE
}  // namespace gr
}  // namespace TestHelpers
