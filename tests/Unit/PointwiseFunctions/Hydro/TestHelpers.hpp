// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <random>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace TestHelpers {
/// \ingroup TestingFrameworkGroup
/// \brief Make random hydro variables which correct physical behavior,
/// e.g. Lorentz factor will be greater or equal than one.
namespace hydro {
template <typename DataType>
Scalar<DataType> random_density(gsl::not_null<std::mt19937*> generator,
                                const DataType& used_for_size) noexcept;

template <typename DataType>
Scalar<DataType> random_lorentz_factor(gsl::not_null<std::mt19937*> generator,
                                       const DataType& used_for_size) noexcept;

template <typename DataType, size_t Dim>
tnsr::I<DataType, Dim> random_velocity(
    gsl::not_null<std::mt19937*> generator,
    const Scalar<DataType>& lorentz_factor,
    const tnsr::ii<DataType, Dim>& spatial_metric) noexcept;

template <typename DataType>
Scalar<DataType> random_temperature(gsl::not_null<std::mt19937*> generator,
                                    const DataType& used_for_size) noexcept;

template <typename DataType>
Scalar<DataType> random_specific_internal_energy(
    gsl::not_null<std::mt19937*> generator,
    const DataType& used_for_size) noexcept;

template <typename DataType>
tnsr::I<DataType, 3> random_magnetic_field(
    gsl::not_null<std::mt19937*> generator, const Scalar<DataType>& pressure,
    const tnsr::ii<DataType, 3>& spatial_metric) noexcept;

template <typename DataType>
Scalar<DataType> random_divergence_cleaning_field(
    gsl::not_null<std::mt19937*> generator,
    const DataType& used_for_size) noexcept;
}  // namespace hydro
}  // namespace TestHelpers
