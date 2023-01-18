// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

/*!
 * \brief Items for assessing truncation error in spectral methods.
 */
namespace PowerMonitors {

/// @{
/*!
 * \ingroup SpectralGroup
 * \brief Returns array of power monitors in each spatial dimension.
 *
 * Computed following Sec. 5.1 of Ref. \cite Szilagyi2014fna.
 * For example, in the x dimension (indexed by \f$ k_0 \f$), we compute
 *
 * \f{align*}{
 *  P_{k_0}[\psi] = \sqrt{ \frac{1}{N_1 N_2}
 *   \sum_{k_1,k_2} \left| C_{k_0,k_1,k_2} \right|^2} ,
 * \f}
 *
 * where \f$ C_{k_0,k_1,k_2}\f$ are the modal coefficients
 * of variable \f$ \psi \f$.
 *
 */
template <size_t Dim>
void power_monitors(gsl::not_null<std::array<DataVector, Dim>*> result,
                    const DataVector& input_data_vector, const Mesh<Dim>& mesh);

template <size_t Dim>
std::array<DataVector, Dim> power_monitors(const DataVector& input_data_vector,
                                           const Mesh<Dim>& mesh);
/// @}

}  // namespace PowerMonitors
