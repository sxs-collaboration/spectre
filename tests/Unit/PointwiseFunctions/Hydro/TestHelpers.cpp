// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/PointwiseFunctions/Hydro/TestHelpers.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"
#include "tests/Utilities/RandomUnitNormal.hpp"

/// \cond
namespace hydro {
namespace TestHelpers {
template <typename DataType>
Scalar<DataType> random_density(const gsl::not_null<std::mt19937*> generator,
                                const DataType& used_for_size) noexcept {
  // 1 g/cm^3 = 1.62e-18 in geometric units
  // Most tests will work fine for any positive density, the exception being
  // tests for primitive inversions in GR hydro and MHD, for which the tolerance
  // would have to be loosened in order to allow for smaller densities.
  constexpr double minimum_density = 1.0e-5;
  constexpr double maximum_density = 1.0e-3;
  std::uniform_real_distribution<> distribution(log(minimum_density),
                                                log(maximum_density));
  return Scalar<DataType>{exp(make_with_random_values<DataType>(
      generator, make_not_null(&distribution), used_for_size))};
}

template <typename DataType>
Scalar<DataType> random_lorentz_factor(
    const gsl::not_null<std::mt19937*> generator,
    const DataType& used_for_size) noexcept {
  std::uniform_real_distribution<> distribution(-10.0, 3.0);
  return Scalar<DataType>{
      1.0 + exp(make_with_random_values<DataType>(
                generator, make_not_null(&distribution), used_for_size))};
}

template <typename DataType, size_t Dim>
tnsr::I<DataType, Dim> random_velocity(
    const gsl::not_null<std::mt19937*> generator,
    const Scalar<DataType>& lorentz_factor,
    const tnsr::ii<DataType, Dim>& spatial_metric) noexcept {
  tnsr::I<DataType, Dim> spatial_velocity =
      random_unit_normal(generator, spatial_metric);
  const DataType v = sqrt(1.0 - 1.0 / square(get(lorentz_factor)));
  for (size_t d = 0; d < Dim; ++d) {
    spatial_velocity.get(d) *= v;
  }
  return spatial_velocity;
}

// temperature in MeV
template <typename DataType>
Scalar<DataType> random_temperature(
    const gsl::not_null<std::mt19937*> generator,
    const DataType& used_for_size) noexcept {
  constexpr double minimum_temperature = 1.0;
  constexpr double maximum_temperature = 50.0;
  std::uniform_real_distribution<> distribution(log(minimum_temperature),
                                                log(maximum_temperature));
  return Scalar<DataType>{exp(make_with_random_values<DataType>(
      generator, make_not_null(&distribution), used_for_size))};
}

template <typename DataType>
Scalar<DataType> random_specific_internal_energy(
    const gsl::not_null<std::mt19937*> generator,
    const DataType& used_for_size) noexcept {
  // assumes Ideal gas with gamma = 4/3
  // For ideal fluid T = (m/k_b)(gamma - 1) epsilon
  // where m = atomic mass unit, k_b = Boltzmann constant
  // m/k_b ~ 933MeV in geometrized units so k_b/[m*(gamma - 1)] ~ 3.21e-3 MeV^-1
  return Scalar<DataType>{3.21e-3 *
                          get(random_temperature(generator, used_for_size))};
}

template <typename DataType>
tnsr::I<DataType, 3> random_magnetic_field(
    const gsl::not_null<std::mt19937*> generator,
    const Scalar<DataType>& pressure,
    const tnsr::ii<DataType, 3>& spatial_metric) noexcept {
  tnsr::I<DataType, 3> magnetic_field =
      random_unit_normal(generator, spatial_metric);
  std::uniform_real_distribution<> distribution(-8.0, 14.0);
  const size_t number_of_points = get_size(get(pressure));
  for (size_t s = 0; s < number_of_points; ++s) {
    // magnitude of B set to vary ratio of magnetic pressure to fluid pressure
    const double B =
        sqrt(get_element(get(pressure), s) * exp(distribution(*generator)));
    get_element(get<0>(magnetic_field), s) *= B;
    get_element(get<1>(magnetic_field), s) *= B;
    get_element(get<2>(magnetic_field), s) *= B;
  }
  return magnetic_field;
}

template <typename DataType>
Scalar<DataType> random_divergence_cleaning_field(
    const gsl::not_null<std::mt19937*> generator,
    const DataType& used_for_size) noexcept {
  std::uniform_real_distribution<> distribution(-10.0, 10.0);
  return make_with_random_values<Scalar<DataType>>(
      generator, make_not_null(&distribution), used_for_size);
}

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

#define INSTANTIATE_SCALARS(_, data)                             \
  template Scalar<DTYPE(data)> random_density(                   \
      const gsl::not_null<std::mt19937*> generator,              \
      const DTYPE(data) & used_for_size) noexcept;               \
  template Scalar<DTYPE(data)> random_lorentz_factor(            \
      const gsl::not_null<std::mt19937*> generator,              \
      const DTYPE(data) & used_for_size) noexcept;               \
  template Scalar<DTYPE(data)> random_temperature(               \
      const gsl::not_null<std::mt19937*> generator,              \
      const DTYPE(data) & used_for_size) noexcept;               \
  template Scalar<DTYPE(data)> random_specific_internal_energy(  \
      const gsl::not_null<std::mt19937*> generator,              \
      const DTYPE(data) & used_for_size) noexcept;               \
  template Scalar<DTYPE(data)> random_divergence_cleaning_field( \
      const gsl::not_null<std::mt19937*> generator,              \
      const DTYPE(data) & used_for_size) noexcept;               \
  template Scalar<DTYPE(data)> random_lapse(                     \
      const gsl::not_null<std::mt19937*> generator,              \
      const DTYPE(data) & used_for_size) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALARS, (double, DataVector))

#define INSTANTIATE_TENSORS(_, data)                                    \
  template tnsr::I<DTYPE(data), DIM(data)> random_velocity(             \
      const gsl::not_null<std::mt19937*> generator,                     \
      const Scalar<DTYPE(data)>& lorentz_factor,                        \
      const tnsr::ii<DTYPE(data), DIM(data)>& spatial_metric) noexcept; \
  template tnsr::I<DTYPE(data), DIM(data)> random_shift(                \
      const gsl::not_null<std::mt19937*> generator,                     \
      const DTYPE(data) & used_for_size) noexcept;                      \
  template tnsr::ii<DTYPE(data), DIM(data)> random_spatial_metric(      \
      const gsl::not_null<std::mt19937*> generator,                     \
      const DTYPE(data) & used_for_size) noexcept;

template tnsr::I<double, 3> random_magnetic_field(
    const gsl::not_null<std::mt19937*> generator,
    const Scalar<double>& pressure,
    const tnsr::ii<double, 3>& spatial_metric) noexcept;
template tnsr::I<DataVector, 3> random_magnetic_field(
    const gsl::not_null<std::mt19937*> generator,
    const Scalar<DataVector>& pressure,
    const tnsr::ii<DataVector, 3>& spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_TENSORS, (double, DataVector), (1, 2, 3))

#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_VECTORS
#undef DIM
#undef DTYPE
}  // namespace TestHelpers
}  // namespace hydro
/// \endcond
