// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ScalarWave {
/// @{
/*!
 * \brief Computes the energy density of the scalar wave system.
 *
 * Below is the function used to calculate the energy density.
 *
 * \f{align*}
 * \epsilon = \frac{1}{2}\left( \Pi^{2} + \abs{\Phi}^{2} \right)
 * \f}
 */
template <size_t SpatialDim>
void energy_density(
    gsl::not_null<Scalar<DataVector>*> result, const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi) noexcept;

template <size_t SpatialDim>
Scalar<DataVector> energy_density(
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi) noexcept;
/// @}
}  // namespace ScalarWave
