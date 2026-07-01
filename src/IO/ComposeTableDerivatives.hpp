// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"

namespace io {

/*!
 * \brief Compute \f$\zeta\f$ analytically from the tabulated free-energy
 * derivatives of a CompOSE 3D EoS.
 *
 * Computes
 * \f[
 * \zeta = \left.\frac{\partial p}{\partial Y_e}\right|_{\rho,\epsilon}
 *       = \left.\frac{\partial(p,\epsilon) / \partial(T,Y_e)}
 *                    {\partial(Y_e,\epsilon) / \partial(T,Y_e)}\right|_{n_b}
 *       = p_{Y_e} - p_T\,\frac{\epsilon_{Y_e}}{\epsilon_T}
 * \f]
 * at fixed baryon density (\f$\rho \propto n_b\f$), where every partial
 * derivative is evaluated analytically from the free energy per baryon
 * \f$\mathcal{F}\f$. With \f$p = n_b^2 \mathcal{F}_{n_b}\f$ and
 * \f$\epsilon = \mathcal{F} - T\mathcal{F}_T\f$ this gives
 * \f{align}{
 * p_{Y_e}      &= n_b^2\,\mathcal{F}_{n_b Y_e}, &
 * p_T          &= n_b^2\,\mathcal{F}_{n_b T}, \\
 * \epsilon_{Y_e} &= \mathcal{F}_{Y_e} - T\,\mathcal{F}_{T Y_e}, &
 * \epsilon_T     &= -T\,\mathcal{F}_{T T}.
 * \f}
 * The neutron-mass scaling and rest-mass offset between CompOSE's stored
 * \f$\epsilon\f$ and \f$\mathcal{F} - T\mathcal{F}_T\f$ cancel because only the
 * ratio \f$\epsilon_{Y_e}/\epsilon_T\f$ enters, so the result is in units of
 * MeV/fm\f$^3\f$ per unit \f$Y_e\f$ (matching the tabulated pressure).
 *
 * The arguments `d2f_dt2`, `d2f_dt_dnb`, `d2f_dt_dye`, `d2f_dnb_dye`, and
 * `df_dye` are the CompOSE free-energy derivatives \f$\mathcal{F}_{TT}\f$,
 * \f$\mathcal{F}_{T n_b}\f$, \f$\mathcal{F}_{T Y_e}\f$,
 * \f$\mathcal{F}_{n_b Y_e}\f$, and \f$\mathcal{F}_{Y_e}\f$ (Table 7.3
 * derivative indices 3, 4, 5, 8, and 9). The `number_density_grid` and
 * `temperature_grid` hold the per-node \f$n_b\f$ (in fm\f$^{-3}\f$) and \f$T\f$
 * (in MeV).
 *
 * The CompOSE table is flattened in file order, with \f$Y_e\f$ varying
 * fastest, then \f$n_b\f$, then \f$T\f$: idx = (iT * nN + in) * nYe + iYe.
 */
DataVector compute_zeta_from_free_energy_derivatives(
    const DataVector& d2f_dt2, const DataVector& d2f_dt_dnb,
    const DataVector& d2f_dt_dye, const DataVector& d2f_dnb_dye,
    const DataVector& df_dye, const std::vector<double>& number_density_grid,
    const std::vector<double>& temperature_grid, size_t number_density_points,
    size_t temperature_points, size_t electron_fraction_points);

}  // namespace io
