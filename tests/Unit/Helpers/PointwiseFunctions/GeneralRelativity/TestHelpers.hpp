// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions useful for testing general relativity

#pragma once

#include <cstddef>
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
/// \brief Make random GR variables which correct physical behavior,
/// e.g. spatial metric will be positive definite
namespace gr {
template <typename DataType>
Scalar<DataType> random_lapse(gsl::not_null<std::mt19937*> generator,
                              const DataType& used_for_size);

template <size_t Dim, typename DataType>
tnsr::I<DataType, Dim> random_shift(gsl::not_null<std::mt19937*> generator,
                                    const DataType& used_for_size);

template <size_t Dim, typename DataType, typename Fr = Frame::Inertial>
tnsr::ii<DataType, Dim, Fr> random_spatial_metric(
    gsl::not_null<std::mt19937*> generator, const DataType& used_for_size);
}  // namespace gr
}  // namespace TestHelpers
