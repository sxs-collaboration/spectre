// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

#include "NumericalAlgorithms/FiniteDifference/Reconstruct.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Direction;
template <size_t Dim>
class Index;
/// \endcond

namespace fd::reconstruction {
namespace detail {
template <size_t NonlinearWeightExponent>
struct AoWeno53Reconstructor {
  SPECTRE_ALWAYS_INLINE static std::array<double, 2> pointwise(
      const double* const u, const int stride, const double gamma_hi,
      const double gamma_lo, const double epsilon) noexcept {
    ASSERT(gamma_hi <= 1.0 and gamma_hi >= 0.0,
           "gamma_hi must be in [0.0, 1.0] but is " << gamma_hi);
    ASSERT(gamma_lo <= 1.0 and gamma_lo >= 0.0,
           "gamma_lo must be in [0.0, 1.0] but is " << gamma_lo);
    ASSERT(epsilon > 0.0,
           "epsilon must be greater than zero but is " << epsilon);

    const std::array moments_sr3_1{
        1.041666666666666 * u[0] - 0.08333333333333333 * u[-stride] +
            0.04166666666666666 * u[-2 * stride],
        0.5 * u[-2 * stride] - 2.0 * u[-stride] + 1.5 * u[0],
        0.5 * u[-2 * stride] - u[-stride] + 0.5 * u[0]};
    const std::array moments_sr3_2{0.04166666666666666 * u[stride] +
                                       0.9166666666666666 * u[0] +
                                       0.04166666666666666 * u[-stride],
                                   0.5 * (u[stride] - u[-stride]),
                                   0.5 * u[-stride] - u[0] + 0.5 * u[stride]};
    const std::array moments_sr3_3{
        0.04166666666666666 * u[2 * stride] - 0.08333333333333333 * u[stride] +
            1.04166666666666666 * u[0],
        -1.5 * u[0] + 2.0 * u[stride] - 0.5 * u[2 * stride],
        0.5 * u[0] - u[stride] + 0.5 * u[2 * stride]};
    const std::array moments_sr5{-2.95138888888888881e-03 * u[-2 * stride] +
                                     5.34722222222222196e-02 * u[-stride] +
                                     8.98958333333333304e-01 * u[0] +
                                     5.34722222222222196e-02 * u[stride] +
                                     -2.95138888888888881e-03 * u[2 * stride],
                                 7.08333333333333315e-02 * u[-2 * stride] +
                                     -6.41666666666666718e-01 * u[-stride] +
                                     6.41666666666666718e-01 * u[stride] +
                                     -7.08333333333333315e-02 * u[2 * stride],
                                 -3.27380952380952397e-02 * u[-2 * stride] +
                                     6.30952380952380931e-01 * u[-stride] +
                                     -1.19642857142857140e+00 * u[0] +
                                     6.30952380952380931e-01 * u[stride] +
                                     -3.27380952380952397e-02 * u[2 * stride],
                                 -8.33333333333333287e-02 * u[-2 * stride] +
                                     1.66666666666666657e-01 * u[-stride] +
                                     -1.66666666666666657e-01 * u[stride] +
                                     8.33333333333333287e-02 * u[2 * stride],
                                 4.16666666666666644e-02 * u[-2 * stride] +
                                     -1.66666666666666657e-01 * u[-stride] +
                                     2.50000000000000000e-01 * u[0] +
                                     -1.66666666666666657e-01 * u[stride] +
                                     4.16666666666666644e-02 * u[2 * stride]};

    // These are the "first alternative" oscillation indicators
    constexpr double beta_r3_factor = 37.0 / 3.0;
    const std::array beta_r3{
        square(moments_sr3_1[1]) + beta_r3_factor * square(moments_sr3_1[2]),
        square(moments_sr3_2[1]) + beta_r3_factor * square(moments_sr3_2[2]),
        square(moments_sr3_3[1]) + beta_r3_factor * square(moments_sr3_3[2])};
    const double beta_sr5 = square(moments_sr5[1]) +
                            61.0 / 5.0 * moments_sr5[1] * moments_sr5[3] +
                            37.0 / 3.0 * square(moments_sr5[2]) +
                            1538.0 / 7.0 * moments_sr5[2] * moments_sr5[4] +
                            8973.0 / 50.0 * square(moments_sr5[3]) +
                            167158.0 / 49.0 * square(moments_sr5[4]);

    // Compute linear and normalized nonlinear weights
    const std::array linear_weights{
        gamma_hi, 0.5 * (1.0 - gamma_hi) * (1.0 - gamma_lo),
        (1.0 - gamma_hi) * gamma_lo, 0.5 * (1.0 - gamma_hi) * (1.0 - gamma_lo)};
    std::array nonlinear_weights{
        linear_weights[0] / pow<NonlinearWeightExponent>(beta_sr5 + epsilon),
        linear_weights[1] / pow<NonlinearWeightExponent>(beta_r3[0] + epsilon),
        linear_weights[2] / pow<NonlinearWeightExponent>(beta_r3[1] + epsilon),
        linear_weights[3] / pow<NonlinearWeightExponent>(beta_r3[2] + epsilon)};
    const double normalization = nonlinear_weights[0] + nonlinear_weights[1] +
                                 nonlinear_weights[2] + nonlinear_weights[3];
    for (double& nw : nonlinear_weights) {
      nw /= normalization;
    }

    const std::array<double, 5> moments{
        {nonlinear_weights[0] / linear_weights[0] *
                 (moments_sr5[0] - linear_weights[1] * moments_sr3_1[0] -
                  linear_weights[2] * moments_sr3_2[0] -
                  linear_weights[3] * moments_sr3_3[0]) +
             nonlinear_weights[1] * moments_sr3_1[0] +
             nonlinear_weights[2] * moments_sr3_2[0] +
             nonlinear_weights[3] * moments_sr3_3[0],
         nonlinear_weights[0] / linear_weights[0] *
                 (moments_sr5[1] - linear_weights[1] * moments_sr3_1[1] -
                  linear_weights[2] * moments_sr3_2[1] -
                  linear_weights[3] * moments_sr3_3[1]) +
             nonlinear_weights[1] * moments_sr3_1[1] +
             nonlinear_weights[2] * moments_sr3_2[1] +
             nonlinear_weights[3] * moments_sr3_3[1],
         nonlinear_weights[0] / linear_weights[0] *
                 (moments_sr5[2] - linear_weights[1] * moments_sr3_1[2] -
                  linear_weights[2] * moments_sr3_2[2] -
                  linear_weights[3] * moments_sr3_3[2]) +
             nonlinear_weights[1] * moments_sr3_1[2] +
             nonlinear_weights[2] * moments_sr3_2[2] +
             nonlinear_weights[3] * moments_sr3_3[2],
         nonlinear_weights[0] / linear_weights[0] * moments_sr5[3],
         nonlinear_weights[0] / linear_weights[0] * moments_sr5[4]}};

    // The polynomial values (excluding the L_0(x)) at the lower/upper faces.
    // The polynomials are (x in [-0.5,0.5]):
    // L0(x)=1.0 (not in array)
    // L1(x)=x
    // L2(x)=x^2-1/12
    // L3(x)=x^3 - (3/20) x
    // L4(x)=x^4 - (3/14) x^2 + 3/560
    const std::array polys_at_plus_half{0.5, 0.16666666666666666, 0.05,
                                        0.014285714285714289};

    return {{moments[0] - polys_at_plus_half[0] * moments[1] +
                 polys_at_plus_half[1] * moments[2] -
                 polys_at_plus_half[2] * moments[3] +
                 polys_at_plus_half[3] * moments[4],
             moments[0] + polys_at_plus_half[0] * moments[1] +
                 polys_at_plus_half[1] * moments[2] +
                 polys_at_plus_half[2] * moments[3] +
                 polys_at_plus_half[3] * moments[4]}};
  }

  SPECTRE_ALWAYS_INLINE static constexpr size_t stencil_width() noexcept {
    return 5;
  }
};
}  // namespace detail

/*!
 * \ingroup FiniteDifferenceGroup
 * \brief Performs an adaptive order (AO) WENO reconstruction using a single 5th
 * order stencil and a 3rd-order CWENO scheme.
 *
 * The AO-WENO(5,3) scheme is based on the scheme presented in
 * \cite Balsara2016780 but adjusted to do reconstruction on variables instead
 * of fluxes. The Legendre basis functions on the domain \f$\xi\in[-1/2,1/2]\f$
 * are given by:
 *
 * \f{align*}{
 * L_0(\xi) &= 1 \\
 * L_1(\xi) &= \xi \\
 * L_2(\xi) &= \xi^2-\frac{1}{12} \\
 * L_3(\xi) &= \xi^3 - \frac{3}{20} \xi \\
 * L_4(\xi) &= \xi^4 - \frac{3}{14} \xi^2 + \frac{3}{560}
 * \f}
 *
 * The oscillation indicators are given by
 *
 * \f{align*}{
 * \beta_l = \sum_{k=1}^{p}\sum_{m=1}^{p}\sigma_{km} u_{k,l} u_{m,l}
 * \f}
 *
 * where \f$p\f$ is the maximum degree of the basis function used for the
 * stencil \f$l\f$, and
 *
 * \f{align*}{
 * \sigma_{km}=\sum_{i=1}^{p}\int_{-1/2}^{1/2}
 * \frac{d^i L_k(\xi)}{d\xi^i}\frac{d^i L_m(\xi)}{d\xi^i}d\xi
 * \f}
 *
 * We write the 3rd-order reconstructed polynomial \f$P^{r3}_l(\xi)\f$
 * associated with the stencil \f$S^{r3}_l\f$ as
 *
 * \f{align*}{
 * P^{r3}_l(\xi) = u_0 +u_{\xi} L_1(\xi) + u_{\xi 2} L_2(\xi)
 * \f}
 *
 * For the stencil \f$S^{r3}_1\f$ we get
 *
 * \f{align*}{
 *  u_0 &= \frac{25}{24} u_{j} -\frac{1}{12} u_{j-1} + \frac{1}{24} u_{j-2} \\
 *  u_{\xi} &= \frac{1}{2}u_{j-2} - 2 u_{j-1} + \frac{3}{2} u_j \\
 *  u_{\xi2} &= \frac{1}{2} u_{j-2} - u_{j-1} + \frac{1}{2} u_j
 * \f}
 *
 * For the stencil \f$S^{r3}_2\f$ we get
 *
 * \f{align*}{
 *  u_0 &= \frac{1}{24} u_{j-1} + \frac{11}{12} u_{j} + \frac{1}{24} u_{j+1} \\
 *  u_{\xi} &= \frac{1}{2}(u_{j+1} - u_{j-1}) \\
 *  u_{\xi2} &= \frac{1}{2} u_{j-1} - u_{j} + \frac{1}{2} u_{j+1}
 * \f}
 *
 * For the stencil \f$S^{r3}_3\f$ we get
 *
 * \f{align*}{
 *  u_0 &= \frac{25}{24} u_{j} -\frac{1}{12} u_{j+1} + \frac{1}{24} u_{j+2} \\
 *  u_{\xi} &= -\frac{1}{2}u_{j+2} + 2 u_{j+1} - \frac{3}{2} u_j \\
 *  u_{\xi2} &= \frac{1}{2} u_{j+2} - u_{j+1} + \frac{1}{2} u_j
 * \f}
 *
 * The oscillation indicator for the 3rd-order stencils is given by
 *
 * \f{align*}{
 *  \beta^{r3}_l = \left(u_{\xi}\right)^2+\frac{13}{3}\left(u_{\xi2}\right)^2
 * \f}
 *
 * We write the 5th-order reconstructed polynomial \f$P^{r5}(\xi)\f$ as
 *
 * \f{align*}{
 * P^{r5}_l(\xi) = u_0 +u_{\xi} L_1(\xi) + u_{\xi 2} L_2(\xi)
 *  + u_{\xi3} L_3(\xi) + u_{\xi4} L_4(\xi)
 * \f}
 *
 * with
 *
 * \f{align*}{
 * u_0 &= -\frac{17}{5760} u_{j-2} + \frac{77}{1440} u_{j-1} +
 *        \frac{863}{960} u_{j} + \frac{77}{1440} u_{j+1} -
 *        \frac{17}{5760} u_{j+2} \\
 * u_{\xi} &= \frac{17}{240} u_{j-2} - \frac{77}{120} u_{j-1} +
 *           \frac{77}{120} u_{j+1} - \frac{17}{240} u_{j+2} \\
 * u_{\xi2} &= -\frac{11}{336} u_{j-2} + \frac{53}{84} u_{j-1} -
 *            \frac{67}{56} u_{j} + \frac{53}{84} u_{j+1} -
 *            \frac{11}{336} u_{j+2} \\
 * u_{\xi3} &= -\frac{1}{12} u_{j-2} + \frac{1}{6} u_{j-1}
 *            -\frac{1}{6} u_{j+1} + \frac{1}{12} u_{j+2} \\
 * u_{\xi4} &= \frac{1}{24} u_{j-2} -\frac{1}{6} u_{j-1}
 *           +\frac{1}{4} u_{j} - \frac{1}{6}u_{j+1}+\frac{1}{24}u_{j+2}
 * \f}
 *
 * The oscillation indicator is given by
 *
 * \f{align*}{
 *  \beta^{r5}&=\left(u_{\xi}+\frac{1}{10}u_{\xi3}\right)^2 \\
 *             &+\frac{13}{3}\left(u_{\xi2}+\frac{123}{455}u_{\xi4}\right)^2\\
 *             &+\frac{781}{20}(u_{\xi3})^2+\frac{1421461}{2275}(u_{\xi4})^2
 * \f}
 *
 * There are two linear weights \f$\gamma_{\mathrm{hi}}\f$ and
 * \f$\gamma_{\mathrm{lo}}\f$. \f$\gamma_{\mathrm{hi}}\f$ controls how much of
 * the 5th-order stencil is used in smooth regions, while
 * \f$\gamma_{\mathrm{lo}}\f$ controls the linear weight of the central
 * 3rd-order stencil. For larger \f$\gamma_{\mathrm{lo}}\f$, the 3rd-order
 * method is more centrally weighted. The linear weights for the stencils
 * are given by
 *
 * \f{align*}{
 *  \gamma^{r5}&=\gamma_{\mathrm{hi}} \\
 *  \gamma^{r3}_1& = (1-\gamma_{\mathrm{hi}})(1-\gamma_{\mathrm{lo}})/2 \\
 *  \gamma^{r3}_2& = (1-\gamma_{\mathrm{hi}})\gamma_{\mathrm{lo}} \\
 *  \gamma^{r3}_3& = (1-\gamma_{\mathrm{hi}})(1-\gamma_{\mathrm{lo}})/2
 * \f}
 *
 * We use the standard nonlinear weights instead of the "Z" weights of
 * \cite Borges20083191
 *
 * \f{align*}{
 *  w^{r5}&=\frac{\gamma^{r5}}{(\beta^{r5}+\epsilon)^q} \\
 *  w^{r3}_l&=\frac{\gamma^{r3}_l}{(\beta^{r3}_l+\epsilon)^q}
 * \f}
 *
 * where \f$\epsilon\f$ is a small number used to avoid division by zero. The
 * normalized nonlinear weights are denoted by \f$\bar{w}^{r5}\f$ and
 * \f$\bar{w}_l^{r3}\f$. The final reconstructed polynomial \f$P(\xi)\f$ is
 * given by
 *
 * \f{align*}{
 *  P(\xi)&=\frac{\bar{w}^{r5}}{\gamma^{r5}}
 *   \left(P^{r5}(\xi)-\gamma^{r3}_1P^{r3}_1(\xi)
 *         -\gamma^{r3}_2P^{r3}_2(\xi)-\gamma^{r3}_3P^{r3}_3(\xi)\right) \\
 *  &+\bar{w}^{r3}_1P^{r3}_1(\xi)+\bar{w}^{r3}_2P^{r3}_2(\xi)
 *  +\bar{w}^{r3}_3P^{r3}_3(\xi)
 * \f}
 *
 * The weights \f$\gamma_{\mathrm{hi}}\f$ and \f$\gamma_{\mathrm{lo}}\f$
 * are typically chosen to be in the range \f$[0.85,0.95]\f$.
 *
 * ### First alternative oscillation indicators
 *
 * Instead of integrating over just the cell, we can instead integrate the basis
 * functions over the entire fit interval, \f$[-5/2,5/2]\f$. Using this interval
 * for the \f$S^{r3}_l\f$ and the \f$S^{r5}\f$ stencils we get
 *
 * \f{align*}{
 *  \beta^{r3}_l&=(u_{\xi})^2 + \frac{37}{3} (u_{\xi2})^2 \\
 *  \beta^{r5}  &=\left(u_{\xi}+\frac{61}{10}u_{\xi3}\right)^2
 *   + \frac{569}{4}u_{\xi3}^2 \\
 *  &+ \frac{1}{8190742}\left(5383 u_{\xi2} + 167158 u_{\xi4}\right)^2
 *  +\frac{4410763}{501474}(u_{\xi2})^2 \\
 * &=u_{\xi}^2 + \frac{61}{5}u_{\xi}u_{\xi3} + \frac{37}{3}u_{\xi2}^2
 *  + \frac{1538}{7}u_{\xi2}u_{\xi4} \\
 * &+ \frac{8973}{50}u_{\xi3}^2 + \frac{167158}{49}u_{\xi4}^2
 * \f}
 *
 * Note that the indicator is manifestly non-negative, a required property of
 * oscillation indicators. These indicators weight high modes more, which means
 * the scheme is more sensitive to high-frequency features in the solution.
 *
 * \note currently it is the alternative indicators that are used. However, an
 * option to control which are used can readily be added, probably best done
 * as a template parameter with `if constexpr` to avoid conditionals inside
 * tight loops.
 */
template <size_t NonlinearWeightExponent, size_t Dim>
void aoweno_53(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_upper_side_of_face_vars,
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_lower_side_of_face_vars,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Index<Dim>& volume_extents, const size_t number_of_variables,
    const double gamma_hi, const double gamma_lo,
    const double epsilon) noexcept {
  detail::reconstruct<detail::AoWeno53Reconstructor<NonlinearWeightExponent>>(
      reconstructed_upper_side_of_face_vars,
      reconstructed_lower_side_of_face_vars, volume_vars, ghost_cell_vars,
      volume_extents, number_of_variables, gamma_hi, gamma_lo, epsilon);
}

/*!
 * \brief Returns function pointers to the `aoweno_53` function, lower neighbor
 * reconstruction, and upper neighbor reconstruction.
 *
 * This is useful for controlling template parameters like the
 * `NonlinearWeightExponent` from an input file by setting a function pointer.
 * Note that the reason the reconstruction functions instead of say the
 * `pointwise` member function is returned is to avoid function pointers inside
 * tight loops.
 */
template <size_t Dim>
std::tuple<
    void (*)(gsl::not_null<std::array<gsl::span<double>, Dim>*>,
             gsl::not_null<std::array<gsl::span<double>, Dim>*>,
             const gsl::span<const double>&,
             const DirectionMap<Dim, gsl::span<const double>>&,
             const Index<Dim>&, size_t, double, double, double) noexcept,
    void (*)(gsl::not_null<DataVector*>, const DataVector&, const DataVector&,
             const Index<Dim>&, const Index<Dim>&, const Direction<Dim>&,
             const double&, const double&, const double&) noexcept,
    void (*)(gsl::not_null<DataVector*>, const DataVector&, const DataVector&,
             const Index<Dim>&, const Index<Dim>&, const Direction<Dim>&,
             const double&, const double&, const double&) noexcept>
aoweno_53_function_pointers(size_t nonlinear_weight_exponent) noexcept;
}  // namespace fd::reconstruction
