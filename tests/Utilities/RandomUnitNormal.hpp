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

/// \ingroup TestingFrameworkGroup
/// \brief Make a random unit normal vector at each element of `DataType`.
template <typename DataType>
tnsr::I<DataType, 1> random_unit_normal(
    gsl::not_null<std::mt19937*> generator,
    const tnsr::ii<DataType, 1>& spatial_metric) noexcept;

template <typename DataType>
tnsr::I<DataType, 2> random_unit_normal(
    gsl::not_null<std::mt19937*> generator,
    const tnsr::ii<DataType, 2>& spatial_metric) noexcept;

template <typename DataType>
tnsr::I<DataType, 3> random_unit_normal(
    gsl::not_null<std::mt19937*> generator,
    const tnsr::ii<DataType, 3>& spatial_metric) noexcept;
