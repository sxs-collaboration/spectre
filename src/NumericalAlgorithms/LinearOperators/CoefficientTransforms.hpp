// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
class ComplexDataVector;
class ComplexModalVector;
class DataVector;
template <size_t Dim>
class Mesh;
class ModalVector;

namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

/// @{
/*!
 * \ingroup SpectralGroup
 * \brief Compute the modal coefficients from the nodal coefficients
 *
 * \see Spectral::nodal_to_modal_matrix
 */
template <size_t Dim>
void to_modal_coefficients(
    gsl::not_null<ComplexModalVector*> modal_coefficients,
    const ComplexDataVector& nodal_coefficients, const Mesh<Dim>& mesh);

// overload provided instead of templating so that the most common case of
// transforming from `DataVector` to `ModalVector` does not require additional
// `make_not_null`s
template <size_t Dim>
void to_modal_coefficients(gsl::not_null<ModalVector*> modal_coefficients,
                           const DataVector& nodal_coefficients,
                           const Mesh<Dim>& mesh);

template <size_t Dim>
ModalVector to_modal_coefficients(const DataVector& nodal_coefficients,
                                  const Mesh<Dim>& mesh);

template <size_t Dim>
ComplexModalVector to_modal_coefficients(
    const ComplexDataVector& nodal_coefficients, const Mesh<Dim>& mesh);
/// @}

/// @{
/*!
 * \ingroup SpectralGroup
 * \brief Compute the nodal coefficients from the modal coefficients
 *
 * \see Spectral::modal_to_nodal_matrix
 */
template <size_t Dim>
void to_nodal_coefficients(gsl::not_null<ComplexDataVector*> nodal_coefficients,
                           const ComplexModalVector& modal_coefficients,
                           const Mesh<Dim>& mesh);

// overload provided instead of templating so that the most common case of
// transforming from `DataVector` to `ModalVector` does not require additional
// `make_not_null`s
template <size_t Dim>
void to_nodal_coefficients(gsl::not_null<DataVector*> nodal_coefficients,
                           const ModalVector& modal_coefficients,
                           const Mesh<Dim>& mesh);

template <size_t Dim>
DataVector to_nodal_coefficients(const ModalVector& modal_coefficients,
                                 const Mesh<Dim>& mesh);

template <size_t Dim>
ComplexDataVector to_nodal_coefficients(
    const ComplexModalVector& modal_coefficients, const Mesh<Dim>& mesh);
/// @}
