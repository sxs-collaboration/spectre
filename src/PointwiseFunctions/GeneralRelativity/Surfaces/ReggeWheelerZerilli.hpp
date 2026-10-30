// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "Utilities/Gsl.hpp"

namespace gr::surfaces {

/*!
 * \ingroup SurfacesGroup
 * \brief Flat-background Regge-Wheeler-Zerilli functions and
 * asymptotic strain.
 *
 * \details The strain convention is
 *
 * \f[
 * h(t,\theta,\phi)=h_+(t,\theta,\phi)-i h_\times(t,\theta,\phi)
 *   =\sum_{\ell,m}h_{\ell m}(t)\,{}_{-2}Y_{\ell m}(\theta,\phi).
 * \f]
 *
 * The spin-weighted spherical harmonics use the orthonormal Goldberg
 * convention, and the modes use Goldberg ordering through `l_max`. The
 * `r_times_strain` field stores the leading-order wave-zone relation between
 * the Regge-Wheeler-Zerilli functions and \f$r h_{\ell m}\f$. Here \f$r\f$ is
 * the supplied coordinate extraction radius, which must approach the areal
 * radius in the asymptotic gauge. This relation is not an exact finite-radius
 * strain: in a flat background it omits terms of order \f$1/r\f$ in \f$r
 * h_{\ell m}\f$, and applying the flat-background formula on a background with
 * mass scale \f$M\f$ also omits relative corrections of order \f$M/r\f$.
 *
 * The stored quantity and mode convention match the data written by SpEC to
 * `rh_FiniteRadii_CodeUnits.h5`; the `FiniteRadii` name identifies where the
 * asymptotic formula is evaluated, not an exact finite-radius strain formula.
 */
struct ReggeWheelerZerilli {
  /// Construct an empty result, for example for use as an output argument.
  ReggeWheelerZerilli() = default;

  /*!
   * \brief Construct zero-initialized modes through `l_max`.
   *
   * \param l_max maximum spherical-harmonic degree represented in the result
   */
  explicit ReggeWheelerZerilli(size_t l_max);

  /// Even-parity Zerilli-Moncrief function modes.
  ComplexModalVector phi_plus{};

  /// Odd-parity Regge-Wheeler-Moncrief function modes.
  ComplexModalVector phi_minus{};

  /// Spin-weight \f$-2\f$ modes of the leading-order asymptotic
  /// \f$r(h_+-i h_\times)\f$.
  SpinWeighted<ComplexModalVector, -2> r_times_strain{};
};

/// @{
/*!
 * \ingroup SurfacesGroup
 * \brief Compute the flat-background Moncrief functions and
 * \f$r h_{\ell m}\f$ from metric-perturbation amplitudes.
 *
 * \details Decompose a metric perturbation into odd- and even-parity scalar,
 * vector, and tensor spherical harmonics as
 *
 * \f{align*}{
 * \delta g^{\mathrm{odd}}_{Ab} &= h_b S_A,\\
 * \delta g^{\mathrm{odd}}_{AB} &= 2k^{\mathrm{odd}}S_{AB},\\
 * \delta g^{\mathrm{even}}_{ab} &= H_{ab}Y,\\
 * \delta g^{\mathrm{even}}_{Ab} &= Q_bY_A,\\
 * \delta g^{\mathrm{even}}_{AB}
 *   &= r^2\left(K\hat{g}_{AB}Y+G Y_{AB}\right).
 * \f}
 *
 * Here \f$a,b\in\{t,r\}\f$, \f$A,B\in\{\theta,\phi\}\f$,
 * \f$\hat{g}_{AB}\f$ is the unit-sphere metric, and \f$S_A\f$,
 * \f$S_{AB}\f$, \f$Y_A\f$, and \f$Y_{AB}\f$ are the odd-parity vector,
 * odd-parity tensor, even-parity vector, and even-parity tensor harmonics,
 * respectively. The odd-parity tensor amplitude \f$k^{\mathrm{odd}}\f$
 * cancels from \f$\Phi^-\f$, so it is not an input. The input names
 * \f$h_{rr}\f$, \f$q_r\f$, \f$k\f$, and \f$g\f$ denote the amplitudes
 * \f$H_{rr}\f$, \f$Q_r\f$, \f$K\f$, and \f$G\f$ in this decomposition.
 * Thus the inputs to this function are the modes
 *
 * \f[
 * \left\{h_t,\partial_r h_t,\partial_t h_r,H_{rr},Q_r,K,
 * \partial_r K,G,\partial_r G\right\}_{\ell m}.
 * \f]
 *
 * This gauge-invariant construction is based on Sec. II of
 * \cite Sarbach:2001qq. The explicit flat-background formulas, with the
 * current SpEC sign convention, are Eqs. (8) and (9) of
 * \cite Buchman:2024zsb. They agree with Eq. (18) of \cite Rinne:2008vn for
 * \f$\Phi^-\f$, while Eq. (29) of that reference defines \f$\Phi^+\f$ with
 * the opposite overall sign. For \f$\ell\geq2\f$, this function computes
 *
 * \f{align*}{
 * p_r &= q_r - \frac{r^2}{2}\partial_r g,\\
 * z_r &= h_{rr} - r\partial_r k
 *        - \frac{r\ell(\ell+1)}{2}\partial_r g - \frac{2p_r}{r},\\
 * k_\mathrm{inv} &= k + \frac{\ell(\ell+1)}{2}g - \frac{2p_r}{r},\\
 * \Phi^-_{\ell m}
 *   &= \frac{r(\partial_t h_r-\partial_r h_t)+2h_t}
 *            {(\ell-1)(\ell+2)},\\
 * \Phi^+_{\ell m}
 *   &= \frac{r(2z_r+(\ell-1)(\ell+2)k_\mathrm{inv})}
 *            {(\ell-1)\ell(\ell+1)(\ell+2)},\\
 * r h_{\ell m}
 *   &= \sqrt{(\ell-1)\ell(\ell+1)(\ell+2)}
 *      \left(\Phi^+_{\ell m}+i\Phi^-_{\ell m}\right).
 * \f}
 *
 * The \f$\Phi^-_{\ell m}\f$ and \f$\Phi^+_{\ell m}\f$ quantities are the
 * flat-background Moncrief functions. The last equation is their
 * leading-order relation to strain at large radius, as in Eq. (54) of
 * \cite Buchman:2024zsb and Eq. (83) of \cite Nagar:2005ea. Equation (4.34)
 * of \cite Ruiz:2007yx is equivalent because its functions are
 * normalized as \f$2\Phi^\pm\f$. The returned value does not include
 * subleading \f$1/r\f$ corrections to \f$r h_{\ell m}\f$.
 *
 * All inputs and outputs contain the full set of positive- and negative-m
 * modes in Goldberg ordering and have size `square(l_max + 1)`. Modes with
 * \f$\ell<2\f$ are set to zero.
 *
 * \param rwz_quantities output Moncrief functions and asymptotic strain modes
 * \param h_t odd-parity amplitudes \f$h_{t,\ell m}\f$
 * \param dr_h_t radial derivatives \f$\partial_r h_{t,\ell m}\f$
 * \param dt_h_r time derivatives \f$\partial_t h_{r,\ell m}\f$
 * \param h_rr even-parity amplitudes \f$H_{rr,\ell m}\f$
 * \param q_r even-parity amplitudes \f$Q_{r,\ell m}\f$
 * \param k even-parity amplitudes \f$K_{\ell m}\f$
 * \param dr_k radial derivatives \f$\partial_r K_{\ell m}\f$
 * \param g even-parity amplitudes \f$G_{\ell m}\f$
 * \param dr_g radial derivatives \f$\partial_r G_{\ell m}\f$
 * \param l_max maximum spherical-harmonic degree in the inputs and result
 * \param extraction_radius coordinate extraction radius \f$r\f$
 */
void regge_wheeler_zerilli_moncrief(
    gsl::not_null<ReggeWheelerZerilli*> rwz_quantities,
    const ComplexModalVector& h_t, const ComplexModalVector& dr_h_t,
    const ComplexModalVector& dt_h_r, const ComplexModalVector& h_rr,
    const ComplexModalVector& q_r, const ComplexModalVector& k,
    const ComplexModalVector& dr_k, const ComplexModalVector& g,
    const ComplexModalVector& dr_g, size_t l_max, double extraction_radius);

/// Return-by-value overload
ReggeWheelerZerilli regge_wheeler_zerilli_moncrief(
    const ComplexModalVector& h_t, const ComplexModalVector& dr_h_t,
    const ComplexModalVector& dt_h_r, const ComplexModalVector& h_rr,
    const ComplexModalVector& q_r, const ComplexModalVector& k,
    const ComplexModalVector& dr_k, const ComplexModalVector& g,
    const ComplexModalVector& dr_g, size_t l_max, double extraction_radius);
/// @}

/// @{
/*!
 * \ingroup SurfacesGroup
 * \brief Compute flat-background RWZ quantities from generalized-harmonic
 * variables on a coordinate sphere.
 *
 * \details The supplied generalized-harmonic variables are interpreted as a
 * linear perturbation of Minkowski spacetime. The points in `inertial_coords`
 * must be the collocation points of `ylm_spherepack` on the coordinate sphere
 * with the supplied `center` and `extraction_radius`. The routine decomposes
 * the perturbation into tensor spherical harmonics and evaluates the Moncrief
 * gauge invariants, following the flat-background extraction described in
 * Sec. 2.3 of \cite Rinne:2008vn, with the \f$\Phi^+\f$ sign convention noted
 * in `regge_wheeler_zerilli_moncrief`. `extraction_radius` is the coordinate
 * radius used in this decomposition. Interpreting `r_times_strain` as
 * asymptotic physical strain requires this radius to approach the areal
 * radius in the asymptotic gauge.
 *
 * The angular grid must have
 * `ylm_spherepack.l_max() >= extraction_l_max + 2`. The two extra modes are
 * needed to transform rank-2 Cartesian tensors without truncating the highest
 * requested tensor-harmonic modes. The result contains modes through
 * `extraction_l_max`.
 *
 * As in SpEC's finite-radius RWZ extraction, this linearized calculation uses
 * \f$\partial_t g_{ij}=-\Pi_{ij}\f$ and
 * \f$\partial_r g_{\alpha\beta}=n^i\Phi_{i\alpha\beta}\f$. It assumes the
 * extraction region is sufficiently close to a flat, spherically symmetric
 * background. On a background with mass scale \f$M\f$, the flat-background
 * strain relation in general neglects relative corrections of order
 * \f$M/r\f$, in addition to the subleading wave-zone terms described above.
 *
 * \note This function first computes the modal inputs \f$h_t\f$,
 * \f$\partial_r h_t\f$, \f$\partial_t h_r\f$, \f$H_{rr}\f$, \f$Q_r\f$,
 * \f$K\f$, \f$\partial_r K\f$, \f$G\f$, and \f$\partial_r G\f$ from the
 * generalized-harmonic variables. It then calls
 * `regge_wheeler_zerilli_moncrief` to form the Moncrief functions and
 * asymptotic strain.
 *
 * \param rwz_quantities output Moncrief functions and asymptotic strain modes
 * \param spacetime_metric spacetime metric \f$g_{\alpha\beta}\f$ at the
 * angular collocation points
 * \param pi generalized-harmonic variable \f$\Pi_{\alpha\beta}\f$, with
 * \f$\partial_t g_{ij}=-\Pi_{ij}\f$ in the linearized calculation
 * \param phi generalized-harmonic variable
 * \f$\Phi_{i\alpha\beta}=\partial_i g_{\alpha\beta}\f$
 * \param inertial_coords inertial coordinates of the angular collocation
 * points
 * \param ylm_spherepack spherical-harmonic transform describing the angular
 * grid
 * \param extraction_l_max maximum spherical-harmonic degree in the result
 * \param center coordinate center of the extraction sphere
 * \param extraction_radius coordinate radius \f$r\f$ of the extraction sphere
 */
void regge_wheeler_zerilli_moncrief_from_gh_vars(
    gsl::not_null<ReggeWheelerZerilli*> rwz_quantities,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& phi,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const ylm::Spherepack& ylm_spherepack, size_t extraction_l_max,
    const std::array<double, 3>& center, double extraction_radius);

/// Return-by-value overload
ReggeWheelerZerilli regge_wheeler_zerilli_moncrief_from_gh_vars(
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& phi,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const ylm::Spherepack& ylm_spherepack, size_t extraction_l_max,
    const std::array<double, 3>& center, double extraction_radius);
/// @}

}  // namespace gr::surfaces
