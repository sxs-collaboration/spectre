// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
template <size_t Dim>
class Index;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace evolution::dg::subcell {
/// @{
/*!
 * \brief Compute and add the 2nd-order flux divergence on a Cartesian mesh to
 * the cell-centered time derivatives.
 */
void add_cartesian_flux_divergence(gsl::not_null<DataVector*> dt_var,
                                   double one_over_delta,
                                   const DataVector& inv_jacobian,
                                   const DataVector& boundary_correction,
                                   const Index<1>& subcell_extents,
                                   size_t dimension) noexcept;

void add_cartesian_flux_divergence(gsl::not_null<DataVector*> dt_var,
                                   double one_over_delta,
                                   const DataVector& inv_jacobian,
                                   const DataVector& boundary_correction,
                                   const Index<2>& subcell_extents,
                                   size_t dimension) noexcept;

void add_cartesian_flux_divergence(gsl::not_null<DataVector*> dt_var,
                                   double one_over_delta,
                                   const DataVector& inv_jacobian,
                                   const DataVector& boundary_correction,
                                   const Index<3>& subcell_extents,
                                   size_t dimension) noexcept;
/// @}
}  // namespace evolution::dg::subcell
