// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "NumericalAlgorithms/SphericalHarmonics/TensorYlm.hpp"
#include "Utilities/Gsl.hpp"

namespace ylm::TensorYlm {

/// @{
/*!
 * \brief Transform scalar-Ylm coefficients into tensor-Ylm coefficients.
 *
 * The `scalar_ylm_coefficients` are stored in Spherepack ordering. For each
 * spectral coefficient, the data for the `number_of_offsets` independent
 * offsets are interleaved contiguously, matching the `*_all_offsets` layout
 * used by `ylm::Spherepack`.
 *
 * The tensor structure of `TensorType` determines the tensor-Ylm sectors that
 * are produced, and `coefficient_normalization` selects whether the stored
 * coefficients use Standard or Spherepack normalization.
 */
template <typename TensorType>
void scalar_to_tensor_ylm_coefficients(
    gsl::not_null<TensorType*> result,
    const TensorType& scalar_ylm_coefficients, size_t l_max,
    size_t number_of_offsets,
    CoefficientNormalization coefficient_normalization);

template <typename TensorType>
TensorType scalar_to_tensor_ylm_coefficients(
    const TensorType& scalar_ylm_coefficients, size_t l_max,
    size_t number_of_offsets,
    CoefficientNormalization coefficient_normalization);
/// @}

/// @{
/*!
 * \brief Transform tensor-Ylm coefficients into scalar-Ylm coefficients.
 *
 * The `tensor_ylm_coefficients` and the output `result` use Spherepack
 * coefficient ordering, with `number_of_offsets` interleaved values per
 * spectral coefficient, matching the `*_all_offsets` layout used by
 * `ylm::Spherepack`.
 *
 * The `coefficient_normalization` selects whether the stored coefficients use
 * Standard or Spherepack normalization.
 */
template <typename TensorType>
void tensor_to_scalar_ylm_coefficients(
    gsl::not_null<TensorType*> result,
    const TensorType& tensor_ylm_coefficients, size_t l_max,
    size_t number_of_offsets,
    CoefficientNormalization coefficient_normalization);

template <typename TensorType>
TensorType tensor_to_scalar_ylm_coefficients(
    const TensorType& tensor_ylm_coefficients, size_t l_max,
    size_t number_of_offsets,
    CoefficientNormalization coefficient_normalization);
/// @}

}  // namespace ylm::TensorYlm
