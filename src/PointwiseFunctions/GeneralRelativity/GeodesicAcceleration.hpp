// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {
/// @{

/*!
 * \brief Computes the coordinate geodesic acceleration in the inertial frame.
 *
 * \details The geodesic acceleration in coordinate form is given by
 * \f{equation}
 *  \frac{d^2 x^i}{d t^2} = (v^i \Gamma^0_{00} - \Gamma^i_{00} )  + 2 v^j (v^i
 * \Gamma^0_{j0} - \Gamma^i_{j0} ) + v^j v^k (v^i \Gamma^0_{jk} - \Gamma^i_{jk}
 * ), \f}
 * where \f$v^i\f$ is the coordinate velocity, \f$\Gamma^\mu_{\nu \rho}\f$ are
 * the spacetime Christoffel symbols of the second kind, and all latin indices
 * are spatial.
 */
template <typename DataType, size_t Dim>
void geodesic_acceleration(
    gsl::not_null<tnsr::I<DataType, Dim>*> acceleration,
    const tnsr::I<DataType, Dim>& velocity,
    const tnsr::Abb<DataType, Dim>& christoffel_second_kind);

template <typename DataType, size_t Dim>
tnsr::I<DataType, Dim> geodesic_acceleration(
    const tnsr::I<DataType, Dim>& velocity,
    const tnsr::Abb<DataType, Dim>& christoffel_second_kind);
/// @}

}  // namespace gr
