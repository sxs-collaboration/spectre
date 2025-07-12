// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/VariablesDeclaration.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
/// \endcond

namespace Spectral::filtering {
/*!
 * \brief Filters the radial components of tensors stored within a `Variables`
 * represented by ZernikeB3 basis functions on a filled ball.
 *
 * \details Functions on a filled ball with angular degree $\ell$ behave as
 * $r^\ell$ near the origin, so their radial profile lies in the even or
 * odd parity ZernikeB3 subspace depending on whether $\ell$ is even or odd. The
 * combined spectral space is therefore indexed by $(n_\mathrm{jac}, \ell, m)$,
 * where $n_\mathrm{jac}$ is the radial Jacobi index. The exponential filter
 * weight for mode $(n_\mathrm{jac}, \ell)$ is
 *
 * \f{align*}{
 *   w = \exp\!\left(-\alpha \left(\frac{n_i}{N_r-1}\right)^{2p}\right),
 * \f}
 *
 * where $n_i = \lfloor (\ell + 2 n_\mathrm{jac}) / 2 \rfloor$ and $N_r$ is the
 * number of radial grid points.
 *
 * The mesh must have basis
 * `(ZernikeB3, ZernikeB3, ZernikeB3)` with quadrature
 * `(GaussRadauUpper, Gauss, Equiangular)` and extents
 * `(n_r, l_max+1, 2*l_max+1)`.
 *
 * \see exponential_filter()
 */
template <typename VariablesTags>
void zernike_b3_ball_radial_exponential_filter(
    gsl::not_null<Variables<VariablesTags>*> u, const Mesh<3>& mesh,
    double alpha, unsigned half_power);

/*!
 * \brief Filters the radial components of tensors stored within a `Variables`
 * represented by ZernikeB3 basis functions on a filled ball.
 *
 * \details Overload taking a caller-managed working buffer. Avoids heap
 * allocations when the filter is applied repeatedly (e.g. every volume call
 * inside `Filters::Ball`).
 */
template <typename VariablesTags>
void zernike_b3_ball_radial_exponential_filter(
    gsl::not_null<Variables<VariablesTags>*> u, gsl::not_null<DataVector*> buf,
    const Mesh<3>& mesh, double alpha, unsigned half_power);

}  // namespace Spectral::filtering
