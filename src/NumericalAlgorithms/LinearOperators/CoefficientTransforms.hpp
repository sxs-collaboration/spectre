// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
class ModalVector;

namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

// @{
/*!
 * \ingroup SpectralGroup
 * \brief Compute the modal coefficients from the nodal coefficients
 *
 * \see Spectral::nodal_to_modal_matrix
 */
template <size_t Dim>
void to_modal_coefficients(gsl::not_null<ModalVector*> modal_coefficients,
                           const DataVector& nodal_coefficients,
                           const Mesh<Dim>& mesh) noexcept;

template <size_t Dim>
ModalVector to_modal_coefficients(const DataVector& nodal_coefficients,
                                  const Mesh<Dim>& mesh) noexcept;
// @}

// @{
/*!
 * \ingroup SpectralGroup
 * \brief Compute the nodal coefficients from the modal coefficients
 *
 * \see Spectral::modal_to_nodal_matrix
 */
template <size_t Dim>
void to_nodal_coefficients(gsl::not_null<DataVector*> nodal_coefficients,
                           const ModalVector& modal_coefficients,
                           const Mesh<Dim>& mesh) noexcept;

template <size_t Dim>
DataVector to_nodal_coefficients(const ModalVector& modal_coefficients,
                                 const Mesh<Dim>& mesh) noexcept;
// @}
