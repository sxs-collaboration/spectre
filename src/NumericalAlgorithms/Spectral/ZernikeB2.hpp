// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
/// \endcond

namespace Spectral {

/*!
 * \ingroup SpectralGroup
 * \brief Transform nodal ZernikeB2 data on a disk to modal space.
 *
 * Transforms \p num_components independent disk slices simultaneously. This
 * is the same operation as done in the disk filtering, while cylinder
 * filtering batches more optimally so we choose not to refactor them to use
 * this looped version. The Fourier nodal-to-modal step is applied to all
 * components in a single batched DGEMM while the Zernike nodal-to-modal
 * step loops over components.
 *
 * \param modal Output spectral coefficients.
 *   Size: `num_components * zernike_b2_disk_spectral_size(n_r_max, n_phi/2)`.
 *   Layout: `modal[comp * spectral_size + spec_index]`.
 * \param buf Scratch buffer. Size: `>= 2 * num_components * n_r * n_phi`.
 * \param u Input nodal values.
 *   Size: `num_components * n_r * n_phi`.
 *   Layout: `u[comp * n_r * n_phi + i_r + n_r * j_phi]`.
 * \param n_r Number of radial grid points.
 * \param n_phi Number of azimuthal grid points (must be odd).
 * \param n_r_max Maximum Zernike degree.
 * \param num_components Number of independent disk slices to transform.
 */
void zernike_b2_disk_nodal_to_modal(gsl::not_null<DataVector*> modal,
                                    gsl::not_null<DataVector*> buf,
                                    const DataVector& u, size_t n_r,
                                    size_t n_phi, size_t n_r_max,
                                    size_t num_components = 1);

/// @{
/*!
 * \ingroup SpectralGroup
 * \brief Returns the radial B2 power monitor indexed by radial spectral level
 * \f$\ell = (n+1)/2\f$ for a function on a ZernikeB2 disk or cylinder mesh.
 *
 * \details For functions represented on a filled disk by ZernikeB2 basis
 * functions, the radial and angular spectral spaces are coupled. This function
 * transforms to the combined ZernikeB2 spectral space and groups based on
 * spectral level, meaning all spectral modes \f$(n, m)\f$ satisfying \f$(n+1)/2
 * = \ell\f$ (using integer division), across all \f$m\f$ and both cosine and
 * sine components, are pooled together.
 *
 * The returned DataVector has \f$N_r\f$ entries \f$(\ell = 0, 1, \ldots,
 * N_r - 1)\f$. The \f$\ell\f$-th entry is
 *
 * \f{align*}{
 *   P_\ell[\psi] = \sqrt{ \frac{1}{S_\ell}
 *     \sum_{\substack{n,m: \\ (n+1)/2 = \ell}} \left| c_{n,m} \right|^2 },
 * \f}
 *
 * where \f$c_{n,m}\f$ are the ZernikeB2 spectral coefficients summing over
 * both cosine and sine components for \f$m \geq 1\f$, and \f$S_\ell\f$ is
 * the total number of spectral coefficient slots at level \f$\ell\f$
 * (including slots that are zero).
 *
 * For the 3D (cylinder) overload, coefficients are pooled across all
 * \f$z\f$-slices: \f$S_\ell\f$ is multiplied by the number of \f$z\f$ points.
 */
void b2_power_monitor_radial(gsl::not_null<DataVector*> result,
                             const DataVector& u, const Mesh<2>& mesh);
void b2_power_monitor_radial(gsl::not_null<DataVector*> result,
                             const DataVector& u, const Mesh<3>& mesh);
/// @}

/// @{
/*!
 * \ingroup SpectralGroup
 * \brief Returns the B2 power monitor indexed by azimuthal wavenumber \f$m\f$
 * for a function on a ZernikeB2 disk or cylinder mesh.
 *
 * \details For functions represented on a filled disk by ZernikeB2 basis
 * functions, the radial and angular spectral spaces are coupled. This function
 * transforms to the combined ZernikeB2 spectral space and returns the
 * root-mean-square of the spectral coefficients at each azimuthal wavenumber
 * \f$m = 0, 1, \ldots, M\f$, where \f$M = N_\phi / 2\f$ and \f$N_\phi\f$ is the
 * number of azimuthal grid points.
 *
 * The returned DataVector has \f$M + 1\f$ entries. The \f$m\f$-th entry is
 *
 * \f{align*}{
 *   P_m[\psi] = \sqrt{ \frac{1}{S_m} \sum_n \left| c_{n,m} \right|^2 },
 * \f}
 *
 * where \f$c_{n,m}\f$ are the ZernikeB2 spectral coefficients of \f$\psi\f$
 * at angular wavenumber \f$m\f$ (summing over both cosine and sine components
 * for \f$m \geq 1\f$), and \f$S_m\f$ is the number of terms in the sum.
 *
 * For the 3D (cylinder) overload, coefficients are pooled across all
 * \f$z\f$-slices: \f$S_m\f$ is multiplied by the number of \f$z\f$ points.
 */
void b2_power_monitor_angular(gsl::not_null<DataVector*> result,
                              const DataVector& u, const Mesh<2>& mesh);
void b2_power_monitor_angular(gsl::not_null<DataVector*> result,
                              const DataVector& u, const Mesh<3>& mesh);
/// @}
}  // namespace Spectral
