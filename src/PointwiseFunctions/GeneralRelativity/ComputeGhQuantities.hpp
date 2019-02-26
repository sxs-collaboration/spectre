// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines Functions for calculating spacetime tensors from 3+1 quantities

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace GeneralizedHarmonic {
// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the auxiliary variable \f$\Phi_{iab}\f$ used by the
 * generalized harmonic formulation of Einstein's equations.
 *
 * \details If \f$ N, N^i\f$ and \f$ g_{ij} \f$ are the lapse, shift and spatial
 * metric respectively, then \f$\Phi_{iab} \f$ is computed as
 *
 * \f{align}
 *     \Phi_{ktt} &= - 2 N \partial_k N
 *                 + 2 g_{mn} N^m \partial_k N^n
 *                 + N^m N^n \partial_k g_{mn} \\
 *     \Phi_{kti} &= g_{mi} \partial_k N^m
 *                 + N^m \partial_k g_{mi} \\
 *     \Phi_{kij} &= \partial_k g_{ij}
 * \f}
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void phi(gsl::not_null<tnsr::iaa<DataType, SpatialDim, Frame>*> phi,
         const Scalar<DataType>& lapse,
         const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
         const tnsr::I<DataType, SpatialDim, Frame>& shift,
         const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
         const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
         const tnsr::ijj<DataType, SpatialDim, Frame>&
             deriv_spatial_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::iaa<DataType, SpatialDim, Frame> phi(
    const Scalar<DataType>& lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>&
        deriv_spatial_metric) noexcept;
// @}

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the conjugate momentum \f$\Pi_{ab}\f$ of the spacetime metric
 * \f$ \psi_{ab} \f$.
 *
 * \details If \f$ N, N^i\f$ are the lapse and shift
 * respectively, and \f$ \Phi_{iab} = \partial_i \psi_{ab} \f$ then
 * \f$\Pi_{\mu\nu} = -\frac{1}{N} ( \partial_t \psi_{\mu\nu}  -
 *      N^m \Phi_{m\mu\nu}) \f$ where \f$ \partial_t \psi_{ab} \f$ is computed
 * as
 *
 * \f{align}
 *     \partial_t \psi_{tt} &= - 2 N \partial_t N
 *                 + 2 g_{mn} N^m \partial_t N^n
 *                 + N^m N^n \partial_t g_{mn} \\
 *     \partial_t \psi_{ti} &= g_{mi} \partial_t N^m
 *                 + N^m \partial_t g_{mi} \\
 *     \partial_t \psi_{ij} &= \partial_t g_{ij}
 * \f}
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void pi(gsl::not_null<tnsr::aa<DataType, SpatialDim, Frame>*> pi,
        const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
        const tnsr::I<DataType, SpatialDim, Frame>& shift,
        const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
        const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
        const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
        const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame> pi(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;
// @}

/*!
 * \ingroup GeneralRelativityGroup
 * \brief  Computes generalized harmonic gauge source function.
 * \details If \f$N, N^i, g_{ij}, \Gamma_{ijk}, K\f$ are the lapse, shift,
 * spatial metric, spatial Christoffel symbols, and trace of the extrinsic
 * curvature, then we compute
 * \f{align}
 * H_l &= N^{-2} g_{il}(\partial_t N^i - N^k \partial_k N^i)
 * + N^{-1} \partial_l N - g^{km}\Gamma_{lkm} \\
 * H_0 &= -N^{-1} \partial_t N + N^{-1} N^k\partial_k N + N^k H_k - N K
 * \f}
 * See Eqs. 8 and 9 of \cite Lindblom2005qh
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> gauge_source(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const Scalar<DataType>& trace_extrinsic_curvature,
    const tnsr::i<DataType, SpatialDim, Frame>&
        trace_christoffel_last_indices) noexcept;

/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes extrinsic curvature from generalized harmonic variables
 *        and the spacetime normal vector.
 *
 * \details If \f$ \Pi_{ab} \f$ and \f$ \Phi_{iab} \f$ are the generalized
 * harmonic conjugate momentum and spatial derivative variables, and if
 * \f$t^a\f$ is the spacetime normal vector, then the extrinsic curvature
 * is computed as
 * \f{align}
 *     K_{ij} &= \frac{1}{2} \Pi_{ij} + \Phi_{(ij)a} t^a
 * \f}
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> extrinsic_curvature(
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes spatial derivatives of the spatial metric from
 *        the generalized harmonic spatial derivative variable.
 *
 * \details If \f$ \Phi_{kab} \f$ is the generalized
 * harmonic spatial derivative variable, then the derivatives of the
 * spatial metric are
 * \f[
 *      \partial_k g_{ij} = \Phi_{kij}
 * \f]
 *
 * This quantity is needed for computing spatial Christoffel symbols.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void deriv_spatial_metric(
    gsl::not_null<tnsr::ijj<DataType, SpatialDim, Frame>*> d_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ijj<DataType, SpatialDim, Frame> deriv_spatial_metric(
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;
// @}

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes time derivative of the spatial metric.
 *
 * \details Let the generalized harmonic conjugate momentum and spatial
 * derivative variables be \f$\Pi_{ab} = -t^c \partial_c \psi_{ab} \f$ and
 * \f$\Phi_{iab} = \partial_i \psi_{ab} \f$. As \f$ t_i \equiv 0 \f$. The time
 * derivative of the spatial metric is given by the time derivative of the
 * spatial sector of the spacetime metric, i.e.
 * \f$ \partial_0 g_{ij} = \partial_0 \psi_{ij} \f$.
 *
 * To compute the latter, we use the evolution equation for \f$ \psi_{ij} \f$,
 * c.f. eq.(35) of \cite Lindblom2005qh (with \f$\gamma_1 = -1\f$):
 *
 * \f[
 * \partial_0 \psi_{ab} = - N \Pi_{ab} + N^k \Phi_{kab}
 * \f]
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void time_deriv_of_spatial_metric(
    gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*> dt_spatial_metric,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> time_deriv_of_spatial_metric(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept;
// @}

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes spacetime derivatives of the determinant of spatial metric,
 *        using the generalized harmonic variables, spatial metric, and its
 *        time derivative.
 *
 * \details Using the relation \f$ \partial_a g = g g^{jk} \partial_a g_{jk} \f$
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_deriv_of_det_spatial_metric(
    gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> d4_det_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> spacetime_deriv_of_det_spatial_metric(
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;
// @}

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes spatial derivatives of lapse (N) from the generalized
 *        harmonic variables and spacetime unit normal 1-form.
 *
 * \details If the generalized harmonic conjugate momentum and spatial
 * derivative variables are \f$\Pi_{ab} = -t^c \partial_c \psi_{ab} \f$ and
 * \f$\Phi_{iab} = \partial_i \psi_{ab} \f$, the spatial derivatives of N
 * can be obtained from:
 * \f{align*}
 *  t^a t^b \Phi_{iab} = -\frac{1}{2N} [\partial_i (-N^2 + N_jN^j)-
 *                               2 N^j \partial_i N_j
 *                               + N^j N^k \partial_i g_{jk}]
 *                     = -\frac{2}{N} \partial_i N,
 * \f}
 * since
 * \f[
 * \partial_i (N_jN^j) = 2N^j \partial_i N_j - N^j N^k \partial_i g_{jk}.
 * \f]
 *
 * \f[
 * \Longrightarrow \partial_i N = -(N/2) t^a \Phi_{iab} t^b
 * \f]
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void spatial_deriv_of_lapse(
    gsl::not_null<tnsr::i<DataType, SpatialDim, Frame>*> deriv_lapse,
    const Scalar<DataType>& lapse,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::i<DataType, SpatialDim, Frame> spatial_deriv_of_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;
// @}

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes time derivative of lapse (N) from the generalized
 *        harmonic variables, lapse, shift and the spacetime unit normal 1-form.
 *
 * \details Let the generalized harmonic conjugate momentum and spatial
 * derivative variables be \f$\Pi_{ab} = -t^c \partial_c \psi_{ab} \f$ and
 * \f$\Phi_{iab} = \partial_i \psi_{ab} \f$, and the operator
 * \f$D := \partial_0 - N^k \partial_k \f$. The time derivative of N is then:
 * \f{align*}
 *  \frac{1}{2} N^2 t^a t^b \Pi_{ab} - \frac{1}{2} N N^i t^a t^b \Phi_{iab}
 *  =& \frac{1}{2} N^2 t^a t^b t^c \partial_c \psi_{ab}
 *       - \frac{1}{2} N N^i (-(2/N) \partial_i N) \\
 *  =& \frac{1}{2} N^2 [-(1/N^3) D[g_{jk} N^j N^k - N^2] \\
 *           &- (N^j N^k / N^3)D[g_{jk}] \\
 *           &+ 2 (N^j / N^3) D[g_{jk} N^k] + (2 / N^2)(N^i \partial_i N)] \\
 *  =& \frac{1}{2N} [-D[g_{jk}N^jN^k - N^2] - N^jN^k D[g_{jk}]
 *            + 2N N^k\partial_k N + 2N^j D[g_{jk}N^k]] \\
 *  =& D[N] + N^k\partial_k N \\
 *  =& \partial_0 N
 * \f}
 * where the simplification done for \f$\partial_i N\f$ is used to substitute
 * for the second term (\f$\frac{1}{2} N N^i t^a t^b \Phi_{iab}\f$).
 *
 * Thus,
 * \f[
 *  \partial_0 N = (N/2)(N t^a t^b \Pi_{ab} - N^i t^a t^b \Phi_{iab})
 * \f]
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void time_deriv_of_lapse(
    gsl::not_null<Scalar<DataType>*> dt_lapse, const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> time_deriv_of_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept;
// @}

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes spatial derivatives of the shift vector from
 *        the generalized harmonic and geometric variables
 *
 * \details Spatial derivatives of the shift vector \f$N^i\f$ can be derived
 * from the following steps:
 * \f{align*}
 * \partial_i N^j
 *  =& g^{jl} g_{kl} \partial_i N^k \\
 *  =& g^{jl} (N^k \partial_i g_{lk}
 *             + g_{kl}\partial_i N^k - N^k \partial_i g_{kl}) \\
 *  =& g^{jl} (\partial_i N_l - N^k \partial_i g_{lk}) (\because g^{j0} = 0) \\
 *  =& g^{ja} (\partial_i \psi_{a0} - N^k \partial _i \psi_{ak}) \\
 *  =& N g^{ja} t^b \partial_i \psi_{ab} \\
 *  =& (g^{ja} - t^j t^a) N t^b \Phi_{iab} - 2 t^j \partial_i N \\
 *  =& \psi^{ja} N t^b \Phi_{iab} - 2 t^j \partial_i N \\
 *  =& N (\psi^{ja} + t^j t^a) t^b \Phi_{iab}.
 * \f}
 * where we used the equation from spatial_deriv_of_lapse() for
 * \f$\partial_i N\f$.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void spatial_deriv_of_shift(
    gsl::not_null<tnsr::iJ<DataType, SpatialDim, Frame>*> deriv_shift,
    const Scalar<DataType>& lapse,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::iJ<DataType, SpatialDim, Frame> spatial_deriv_of_shift(
    const Scalar<DataType>& lapse,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;
// @}

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes time derivative of the shift vector from
 *        the generalized harmonic and geometric variables
 *
 * \details The time derivative of \f$ N^i \f$ can be derived from the following
 * steps:
 * \f{align*}
 * \partial_0 N^i
 *  =& g^{ik} \partial_0 (g_{kj} N^j) - N^j g^{ik} \partial_0 g_{kj} \\
 *  =& N g^{ik} t^b \partial_0 \psi_{kb} \\
 *  =& N g^{ik} t^b (\partial_0 - N^j\partial_j) \psi_{kb}
 *                  + N g^{ik} t^b N^j\partial_j \psi_{kb} \\
 *  =& -N^2 t^b\Pi_{kb} g^{ik} + N N^j t^b\Phi_{jkb} g^{ik} \\
 *  =& -N g^{ik} t^b (N \Pi_{kb} - N^j \Phi_{jkb}) \\
 * \f}
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void time_deriv_of_shift(
    gsl::not_null<tnsr::I<DataType, SpatialDim, Frame>*> dt_shift,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::I<DataType, SpatialDim, Frame> time_deriv_of_shift(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept;
// @}

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes time derivative of index lowered shift from generalized
 *        harmonic variables, spatial metric and its time derivative.
 *
 * \details The time derivative of \f$ N_i \f$ is given by:
 * \f{align*}
 *  \partial_0 N_i = g_{ij} \partial_0 N^j + N^j \partial_0 g_{ij}
 * \f}
 * where the first term is obtained from `time_deriv_of_shift()`, and the latter
 * is a user input.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void time_deriv_of_lower_shift(
    gsl::not_null<tnsr::i<DataType, SpatialDim, Frame>*> dt_lower_shift,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::i<DataType, SpatialDim, Frame> time_deriv_of_lower_shift(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept;
// @}

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes spacetime derivatives of the norm of the shift vector.
 *
 * \details The same is computed as:
 * \f{align*}
 * \partial_a (N^i N_i) = (N_i \partial_0 N^i + N^i \partial_0 N_i,
 *                               N_i \partial_j N^i + N^i \partial_j N_i)
 * \f}
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_deriv_of_norm_of_shift(
    gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> d4_norm_of_shift,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> spacetime_deriv_of_norm_of_shift(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept;
// @}
}  // namespace GeneralizedHarmonic
