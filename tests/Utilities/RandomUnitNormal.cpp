// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Utilities/RandomUnitNormal.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

/// \cond
template <typename DataType>
tnsr::I<DataType, 1> random_unit_normal(
    const gsl::not_null<std::mt19937*> generator,
    const tnsr::ii<DataType, 1>& spatial_metric) noexcept {
  std::uniform_int_distribution<> distribution(0, 1);
  auto unit_normal = make_with_value<tnsr::I<DataType, 1>>(
      spatial_metric, 1.0 - 2.0 * distribution(*generator));
  get<0>(unit_normal) /= get(magnitude(unit_normal, spatial_metric));
  return unit_normal;
}

template <typename DataType>
tnsr::I<DataType, 2> random_unit_normal(
    const gsl::not_null<std::mt19937*> generator,
    const tnsr::ii<DataType, 2>& spatial_metric) noexcept {
  std::uniform_real_distribution<> distribution(-M_PI, M_PI);
  // non-const to reuse allocation
  auto phi = make_with_random_values<DataType>(
      generator, make_not_null(&distribution), spatial_metric);
  auto unit_normal = make_with_value<tnsr::I<DataType, 2>>(spatial_metric, 1.0);
  get<0>(unit_normal) *= cos(phi);
  get<1>(unit_normal) *= sin(phi);
  DataType one_over_magnitude = std::move(phi);
  one_over_magnitude = 1.0 / get(magnitude(unit_normal, spatial_metric));
  get<0>(unit_normal) *= one_over_magnitude;
  get<1>(unit_normal) *= one_over_magnitude;
  return unit_normal;
}

template <typename DataType>
tnsr::I<DataType, 3> random_unit_normal(
    const gsl::not_null<std::mt19937*> generator,
    const tnsr::ii<DataType, 3>& spatial_metric) noexcept {
  std::uniform_real_distribution<> nz_distribution(-1.0, 1.0);
  // non-const to reuse allocation
  auto nz = make_with_random_values<DataType>(
      generator, make_not_null(&nz_distribution), spatial_metric);
  const DataType rho = sqrt(1.0 - square(nz));
  std::uniform_real_distribution<> phi_distribution(-M_PI, M_PI);
  const auto phi = make_with_random_values<DataType>(
      generator, make_not_null(&phi_distribution), spatial_metric);
  auto unit_normal = make_with_value<tnsr::I<DataType, 3>>(spatial_metric, 0.0);
  get<0>(unit_normal) = rho * cos(phi);
  get<1>(unit_normal) = rho * sin(phi);
  get<2>(unit_normal) = nz;
  DataType one_over_magnitude = std::move(nz);
  one_over_magnitude = 1.0 / get(magnitude(unit_normal, spatial_metric));
  get<0>(unit_normal) *= one_over_magnitude;
  get<1>(unit_normal) *= one_over_magnitude;
  get<2>(unit_normal) *= one_over_magnitude;
  return unit_normal;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data)                                 \
  template tnsr::I<DTYPE(data), DIM(data)> random_unit_normal( \
      const gsl::not_null<std::mt19937*> generator,            \
      const tnsr::ii<DTYPE(data), DIM(data)>& spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (double, DataVector), (1, 2, 3))

#undef INSTANTIATION
#undef DIM
#undef DTYPE
/// \endcond
