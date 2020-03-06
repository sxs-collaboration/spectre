// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines Functions for calculating spacetime tensors from 3+1 quantities

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

/// \cond
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
namespace domain {
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace Tags
}  // namespace domain
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;
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
 * \brief Computes the time derivative of the spacetime metric from the
 * generalized harmonic quantities \f$\Pi_{a b}\f$, \f$\Phi_{i a b}\f$, and the
 * lapse \f$\alpha\f$ and shift \f$\beta^i\f$.
 *
 * \details Computes the derivative as:
 *
 * \f{align}{
 * \partial_t \psi_{a b} = \beta^i \Phi_{i a b} - \alpha \Pi_{a b}.
 * \f}
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void time_derivative_of_spacetime_metric(
    gsl::not_null<tnsr::aa<DataType, SpatialDim, Frame>*> dt_spacetime_metric,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame> time_derivative_of_spacetime_metric(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;
//@}

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

namespace Tags {
/*!
 * \brief Compute item to get time derivative of the spatial metric from
 *        generalized harmonic and geometric variables
 *
 * \details See `time_deriv_of_spatial_metric()`. Can be retrieved using
 * `gr::Tags::SpatialMetric` wrapped in `Tags::dt`.
 */
template <size_t SpatialDim, typename Frame>
struct TimeDerivSpatialMetricCompute
    : ::Tags::dt<gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<SpatialDim, Frame, DataVector>,
                 Phi<SpatialDim, Frame>, Pi<SpatialDim, Frame>>;
  static constexpr tnsr::ii<DataVector, SpatialDim, Frame> (*function)(
      const Scalar<DataVector>&, const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::aa<DataVector, SpatialDim, Frame>&) =
      &time_deriv_of_spatial_metric<SpatialDim, Frame>;
  using base =
      ::Tags::dt<gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>>;
};

/*!
 * \brief Compute item to get time derivative of lapse (N) from the generalized
 *        harmonic variables, lapse, shift and the spacetime unit normal 1-form.
 *
 * \details See `time_deriv_of_lapse()`. Can be retrieved using
 * `gr::Tags::Lapse` wrapped in `Tags::dt`.
 */
template <size_t SpatialDim, typename Frame>
struct TimeDerivLapseCompute : ::Tags::dt<gr::Tags::Lapse<DataVector>>,
                               db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<SpatialDim, Frame, DataVector>,
                 gr::Tags::SpacetimeNormalVector<SpatialDim, Frame, DataVector>,
                 Phi<SpatialDim, Frame>, Pi<SpatialDim, Frame>>;
  static constexpr Scalar<DataVector> (*function)(
      const Scalar<DataVector>&, const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::A<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::aa<DataVector, SpatialDim, Frame>&) =
      &time_deriv_of_lapse<SpatialDim, Frame>;
  using base = ::Tags::dt<gr::Tags::Lapse<DataVector>>;
};

/*!
 * \brief Compute item to get time derivative of the shift vector from
 *        the generalized harmonic and geometric variables
 *
 * \details See `time_deriv_of_shift()`. Can be retrieved using
 * `gr::Tags::Shift` wrapped in `Tags::dt`.
 */
template <size_t SpatialDim, typename Frame>
struct TimeDerivShiftCompute
    : ::Tags::dt<gr::Tags::Shift<SpatialDim, Frame, DataVector>>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<SpatialDim, Frame, DataVector>,
                 gr::Tags::InverseSpatialMetric<SpatialDim, Frame, DataVector>,
                 gr::Tags::SpacetimeNormalVector<SpatialDim, Frame, DataVector>,
                 Phi<SpatialDim, Frame>, Pi<SpatialDim, Frame>>;
  static constexpr tnsr::I<DataVector, SpatialDim, Frame> (*function)(
      const Scalar<DataVector>&, const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::II<DataVector, SpatialDim, Frame>&,
      const tnsr::A<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::aa<DataVector, SpatialDim, Frame>&) =
      &time_deriv_of_shift<SpatialDim, Frame, DataVector>;
  using base = ::Tags::dt<gr::Tags::Shift<SpatialDim, Frame, DataVector>>;
};

/*!
 * \brief Compute item to get spatial derivatives of the spatial metric from
 *        the generalized harmonic spatial derivative variable.
 *
 * \details See `deriv_spatial_metric()`. Can be retrieved using
 * `gr::Tags::SpatialMetric` wrapped in `::Tags::deriv`.
 */
template <size_t SpatialDim, typename Frame>
struct DerivSpatialMetricCompute
    : ::Tags::deriv<gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>,
                    tmpl::size_t<SpatialDim>, Frame>,
      db::ComputeTag {
  using argument_tags = tmpl::list<Phi<SpatialDim, Frame>>;
  static constexpr tnsr::ijj<DataVector, SpatialDim, Frame> (*function)(
      const tnsr::iaa<DataVector, SpatialDim, Frame>&) =
      &deriv_spatial_metric<SpatialDim, Frame>;
  using base =
      ::Tags::deriv<gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>,
                    tmpl::size_t<SpatialDim>, Frame>;
};

/*!
 * \brief Compute item to get spatial derivatives of lapse from the
 * generalized harmonic variables and spacetime unit normal one-form.
 *
 * \details See `spatial_deriv_of_lapse()`. Can be retrieved using
 * `gr::Tags::Lapse` wrapped in `::Tags::deriv`.
 */
template <size_t SpatialDim, typename Frame>
struct DerivLapseCompute : ::Tags::deriv<gr::Tags::Lapse<DataVector>,
                                         tmpl::size_t<SpatialDim>, Frame>,
                           db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>,
                 gr::Tags::SpacetimeNormalVector<SpatialDim, Frame, DataVector>,
                 Phi<SpatialDim, Frame>>;
  static constexpr tnsr::i<DataVector, SpatialDim, Frame> (*function)(
      const Scalar<DataVector>&, const tnsr::A<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&) =
      &spatial_deriv_of_lapse<SpatialDim, Frame>;
  using base = ::Tags::deriv<gr::Tags::Lapse<DataVector>,
                             tmpl::size_t<SpatialDim>, Frame>;
};

/*!
 * \brief Compute item to get spatial derivatives of the shift vector from
 *        generalized harmonic and geometric variables
 *
 * \details See `spatial_deriv_of_shift()`. Can be retrieved using
 * `gr::Tags::Shift` wrapped in `::Tags::deriv`.
 */
template <size_t SpatialDim, typename Frame>
struct DerivShiftCompute
    : ::Tags::deriv<gr::Tags::Shift<SpatialDim, Frame, DataVector>,
                    tmpl::size_t<SpatialDim>, Frame>,
      db::ComputeTag {
  using argument_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>,
      gr::Tags::InverseSpacetimeMetric<SpatialDim, Frame, DataVector>,
      gr::Tags::SpacetimeNormalVector<SpatialDim, Frame, DataVector>,
      Phi<SpatialDim, Frame>>;
  static constexpr tnsr::iJ<DataVector, SpatialDim, Frame> (*function)(
      const Scalar<DataVector>&, const tnsr::AA<DataVector, SpatialDim, Frame>&,
      const tnsr::A<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&) =
      &spatial_deriv_of_shift<SpatialDim, Frame, DataVector>;
  using base = ::Tags::deriv<gr::Tags::Shift<SpatialDim, Frame, DataVector>,
                             tmpl::size_t<SpatialDim>, Frame>;
};

/*!
 * \brief Compute item for the auxiliary variable \f$\Phi_{iab}\f$ used by the
 * generalized harmonic formulation of Einstein's equations.
 *
 * \details See `phi()`. Can be retrieved using
 * `GeneralizedHarmonic::Tags::Phi`.
 */
template <size_t SpatialDim, typename Frame>
struct PhiCompute : Phi<SpatialDim, Frame>, db::ComputeTag {
  using argument_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<SpatialDim>,
                    Frame>,
      gr::Tags::Shift<SpatialDim, Frame, DataVector>,
      ::Tags::deriv<gr::Tags::Shift<SpatialDim, Frame, DataVector>,
                    tmpl::size_t<SpatialDim>, Frame>,
      gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>,
      ::Tags::deriv<gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>,
                    tmpl::size_t<SpatialDim>, Frame>>;
  static constexpr tnsr::iaa<DataVector, SpatialDim, Frame> (*function)(
      const Scalar<DataVector>&, const tnsr::i<DataVector, SpatialDim, Frame>&,
      const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::iJ<DataVector, SpatialDim, Frame>&,
      const tnsr::ii<DataVector, SpatialDim, Frame>&,
      const tnsr::ijj<DataVector, SpatialDim, Frame>&) =
      &phi<SpatialDim, Frame, DataVector>;
  using base = Phi<SpatialDim, Frame>;
};

/*!
 * \brief Compute item the conjugate momentum \f$\Pi_{ab}\f$ of the spacetime
 * metric \f$ \psi_{ab} \f$.
 *
 * \details See `pi()`. Can be retrieved using `GeneralizedHarmonic::Tags::Pi`.
 */
template <size_t SpatialDim, typename Frame>
struct PiCompute : Pi<SpatialDim, Frame>, db::ComputeTag {
  using argument_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>, ::Tags::dt<gr::Tags::Lapse<DataVector>>,
      gr::Tags::Shift<SpatialDim, Frame, DataVector>,
      ::Tags::dt<gr::Tags::Shift<SpatialDim, Frame, DataVector>>,
      gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>,
      ::Tags::dt<gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>>,
      Phi<SpatialDim, Frame>>;
  static constexpr tnsr::aa<DataVector, SpatialDim, Frame> (*function)(
      const Scalar<DataVector>&, const Scalar<DataVector>&,
      const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::ii<DataVector, SpatialDim, Frame>&,
      const tnsr::ii<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&) =
      &pi<SpatialDim, Frame, DataVector>;
  using base = Pi<SpatialDim, Frame>;
};

/*!
 * \brief Compute item to get extrinsic curvature from generalized harmonic
 * variables and the spacetime normal vector.
 *
 * \details See `extrinsic_curvature()`. Can be retrieved using
 * `gr::Tags::ExtrinsicCurvature`.
 */
template <size_t SpatialDim, typename Frame>
struct ExtrinsicCurvatureCompute
    : gr::Tags::ExtrinsicCurvature<SpatialDim, Frame, DataVector>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::SpacetimeNormalVector<SpatialDim, Frame, DataVector>,
                 Pi<SpatialDim, Frame>, Phi<SpatialDim, Frame>>;
  static constexpr auto function =
      &extrinsic_curvature<SpatialDim, Frame, DataVector>;
  using base = gr::Tags::ExtrinsicCurvature<SpatialDim, Frame, DataVector>;
};

/*!
 * \brief Compute item to get the trace of extrinsic curvature from generalized
 * harmonic variables and the spacetime normal vector.
 *
 * \details See `extrinsic_curvature()` for how the extrinsic curvature
 * \f$ K_{ij}\f$ is computed. Its trace is taken as
 * \f{align}
 *     tr(K) &= g^{ij} K_{ij}.
 * \f}
 *
 * Can be retrieved using `gr::Tags::TraceExtrinsicCurvature`.
 */
template <size_t SpatialDim, typename Frame>
struct TraceExtrinsicCurvatureCompute
    : gr::Tags::TraceExtrinsicCurvature<DataVector>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::ExtrinsicCurvature<SpatialDim, Frame, DataVector>,
                 gr::Tags::InverseSpatialMetric<SpatialDim, Frame, DataVector>>;
  static constexpr Scalar<DataVector> (*function)(
      const tnsr::ii<DataVector, SpatialDim, Frame>&,
      const tnsr::II<DataVector, SpatialDim, Frame>&) = &trace;
  using base = gr::Tags::TraceExtrinsicCurvature<DataVector>;
};

/*!
 * \brief Compute items to compute constraint-damping parameters for a
 * single-BH evolution.
 *
 * \details Can be retrieved using
 * `GeneralizedHarmonic::Tags::ConstraintGamma0`,
 * `GeneralizedHarmonic::Tags::ConstraintGamma1`, and
 * `GeneralizedHarmonic::Tags::ConstraintGamma2`.
 */
template <size_t SpatialDim, typename Frame>
struct ConstraintGamma0Compute : ConstraintGamma0, db::ComputeTag {
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<SpatialDim, Frame>>;
  static auto function(
      const tnsr::I<DataVector, SpatialDim, Frame>& coords) noexcept {
    const DataVector r_squared = get(dot_product(coords, coords));
    Scalar<DataVector> gamma = make_with_value<type>(coords, 0.);
    get(gamma) = 3. * exp(-0.0078125 * r_squared) + 0.001;
    return gamma;
  }
  using base = ConstraintGamma0;
};
/// \copydoc ConstraintGamma0Compute
template <size_t SpatialDim, typename Frame>
struct ConstraintGamma1Compute : ConstraintGamma1, db::ComputeTag {
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<SpatialDim, Frame>>;
  static constexpr auto function(
      const tnsr::I<DataVector, SpatialDim, Frame>& coords) noexcept {
    return make_with_value<type>(coords, -1.);
  }
  using base = ConstraintGamma1;
};
/// \copydoc ConstraintGamma0Compute
template <size_t SpatialDim, typename Frame>
struct ConstraintGamma2Compute : ConstraintGamma2, db::ComputeTag {
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<SpatialDim, Frame>>;
  static auto function(
      const tnsr::I<DataVector, SpatialDim, Frame>& coords) noexcept {
    const DataVector r_squared = get(dot_product(coords, coords));
    Scalar<DataVector> gamma = make_with_value<type>(coords, 0.);
    get(gamma) = exp(-0.0078125 * r_squared) + 0.001;
    return gamma;
  }
  using base = ConstraintGamma2;
};

/*!
 * \brief  Compute item to get the implicit gauge source function from 3 + 1
 * quantities.
 *
 * \details See `gauge_source()`. Can be retrieved using
 * `GeneralizedHarmonic::Tags::GaugeH`.
 */
template <size_t SpatialDim, typename Frame>
struct GaugeHImplicitFrom3p1QuantitiesCompute : GaugeH<SpatialDim, Frame>,
                                                db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>,
                 ::Tags::dt<gr::Tags::Lapse<DataVector>>,
                 ::Tags::deriv<gr::Tags::Lapse<DataVector>,
                               tmpl::size_t<SpatialDim>, Frame>,
                 gr::Tags::Shift<SpatialDim, Frame, DataVector>,
                 ::Tags::dt<gr::Tags::Shift<SpatialDim, Frame, DataVector>>,
                 ::Tags::deriv<gr::Tags::Shift<SpatialDim, Frame, DataVector>,
                               tmpl::size_t<SpatialDim>, Frame>,
                 gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>,
                 gr::Tags::TraceExtrinsicCurvature<DataVector>,
                 gr::Tags::TraceSpatialChristoffelFirstKind<SpatialDim, Frame,
                                                            DataVector>>;
  static constexpr auto function = &gauge_source<SpatialDim, Frame, DataVector>;
  using base = GaugeH<SpatialDim, Frame>;
};

/*!
 * \brief  Compute item to get spacetime derivative of the gauge source function
 * from its spatial and time derivatives.
 *
 * \details Can be retrieved using
 * `GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH`.
 */
template <size_t SpatialDim, typename Frame>
struct SpacetimeDerivGaugeHCompute : SpacetimeDerivGaugeH<SpatialDim, Frame>,
                                     db::ComputeTag {
  using argument_tags = tmpl::list<
      ::Tags::dt<GeneralizedHarmonic::Tags::GaugeH<SpatialDim, Frame>>,
      ::Tags::deriv<GeneralizedHarmonic::Tags::GaugeH<SpatialDim, Frame>,
                    tmpl::size_t<SpatialDim>, Frame>>;
  static constexpr tnsr::ab<DataVector, SpatialDim, Frame> function(
      const tnsr::a<DataVector, SpatialDim, Frame>& time_deriv_gauge_source,
      const tnsr::ia<DataVector, SpatialDim, Frame>& deriv_gauge_source) {
    auto spacetime_deriv_gauge_source =
        make_with_value<tnsr::ab<DataVector, SpatialDim, Frame>>(
            time_deriv_gauge_source, 0.0);
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      spacetime_deriv_gauge_source.get(0, b) = time_deriv_gauge_source.get(b);
      for (size_t a = 1; a < SpatialDim + 1; ++a) {
        spacetime_deriv_gauge_source.get(a, b) =
            deriv_gauge_source.get(a - 1, b);
      }
    }
    return spacetime_deriv_gauge_source;
  }
  using base = SpacetimeDerivGaugeH<SpatialDim, Frame>;
};

/*!
 * \brief Compute item to get the gauge constraint for the
 * generalized harmonic evolution system.
 *
 * \details See `gauge_constraint()`. Can be retrieved using
 * `GeneralizedHarmonic::Tags::GaugeConstraint`.
 */
template <size_t SpatialDim, typename Frame>
struct GaugeConstraintCompute : GaugeConstraint<SpatialDim, Frame>,
                                db::ComputeTag {
  using argument_tags = tmpl::list<
      GaugeH<SpatialDim, Frame>,
      gr::Tags::SpacetimeNormalOneForm<SpatialDim, Frame, DataVector>,
      gr::Tags::SpacetimeNormalVector<SpatialDim, Frame, DataVector>,
      gr::Tags::InverseSpatialMetric<SpatialDim, Frame, DataVector>,
      gr::Tags::InverseSpacetimeMetric<SpatialDim, Frame, DataVector>,
      Pi<SpatialDim, Frame>, Phi<SpatialDim, Frame>>;
  static constexpr tnsr::a<DataVector, SpatialDim, Frame> (*function)(
      const tnsr::a<DataVector, SpatialDim, Frame>&,
      const tnsr::a<DataVector, SpatialDim, Frame>&,
      const tnsr::A<DataVector, SpatialDim, Frame>&,
      const tnsr::II<DataVector, SpatialDim, Frame>&,
      const tnsr::AA<DataVector, SpatialDim, Frame>&,
      const tnsr::aa<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&) =
      &gauge_constraint<SpatialDim, Frame, DataVector>;
  using base = GaugeConstraint<SpatialDim, Frame>;
};

/*!
 * \brief Compute item to get the F-constraint for the
 * generalized harmonic evolution system.
 *
 * \details See `f_constraint()`. Can be retrieved using
 * `GeneralizedHarmonic::Tags::FConstraint`.
 */
template <size_t SpatialDim, typename Frame>
struct FConstraintCompute : FConstraint<SpatialDim, Frame>, db::ComputeTag {
  using argument_tags = tmpl::list<
      GaugeH<SpatialDim, Frame>,
      ::Tags::deriv<GaugeH<SpatialDim, Frame>, tmpl::size_t<SpatialDim>, Frame>,
      gr::Tags::SpacetimeNormalOneForm<SpatialDim, Frame, DataVector>,
      gr::Tags::SpacetimeNormalVector<SpatialDim, Frame, DataVector>,
      gr::Tags::InverseSpatialMetric<SpatialDim, Frame, DataVector>,
      gr::Tags::InverseSpacetimeMetric<SpatialDim, Frame, DataVector>,
      Pi<SpatialDim, Frame>, Phi<SpatialDim, Frame>,
      ::Tags::deriv<Pi<SpatialDim, Frame>, tmpl::size_t<SpatialDim>, Frame>,
      ::Tags::deriv<Phi<SpatialDim, Frame>, tmpl::size_t<SpatialDim>, Frame>,
      ConstraintGamma2, ThreeIndexConstraint<SpatialDim, Frame>>;
  static constexpr tnsr::a<DataVector, SpatialDim, Frame> (*function)(
      const tnsr::a<DataVector, SpatialDim, Frame>&,
      const tnsr::ia<DataVector, SpatialDim, Frame>&,
      const tnsr::a<DataVector, SpatialDim, Frame>&,
      const tnsr::A<DataVector, SpatialDim, Frame>&,
      const tnsr::II<DataVector, SpatialDim, Frame>&,
      const tnsr::AA<DataVector, SpatialDim, Frame>&,
      const tnsr::aa<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::ijaa<DataVector, SpatialDim, Frame>&,
      const Scalar<DataVector>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&) =
      &f_constraint<SpatialDim, Frame, DataVector>;
  using base = FConstraint<SpatialDim, Frame>;
};

/*!
 * \brief Compute item to get the two-index constraint for the
 * generalized harmonic evolution system.
 *
 * \details See `two_index_constraint()`. Can be retrieved using
 * `GeneralizedHarmonic::Tags::TwoIndexConstraint`.
 */
template <size_t SpatialDim, typename Frame>
struct TwoIndexConstraintCompute : TwoIndexConstraint<SpatialDim, Frame>,
                                   db::ComputeTag {
  using argument_tags = tmpl::list<
      ::Tags::deriv<GaugeH<SpatialDim, Frame>, tmpl::size_t<SpatialDim>, Frame>,
      gr::Tags::SpacetimeNormalOneForm<SpatialDim, Frame, DataVector>,
      gr::Tags::SpacetimeNormalVector<SpatialDim, Frame, DataVector>,
      gr::Tags::InverseSpatialMetric<SpatialDim, Frame, DataVector>,
      gr::Tags::InverseSpacetimeMetric<SpatialDim, Frame, DataVector>,
      Pi<SpatialDim, Frame>, Phi<SpatialDim, Frame>,
      ::Tags::deriv<Pi<SpatialDim, Frame>, tmpl::size_t<SpatialDim>, Frame>,
      ::Tags::deriv<Phi<SpatialDim, Frame>, tmpl::size_t<SpatialDim>, Frame>,
      ConstraintGamma2, ThreeIndexConstraint<SpatialDim, Frame>>;
  static constexpr tnsr::ia<DataVector, SpatialDim, Frame> (*function)(
      const tnsr::ia<DataVector, SpatialDim, Frame>&,
      const tnsr::a<DataVector, SpatialDim, Frame>&,
      const tnsr::A<DataVector, SpatialDim, Frame>&,
      const tnsr::II<DataVector, SpatialDim, Frame>&,
      const tnsr::AA<DataVector, SpatialDim, Frame>&,
      const tnsr::aa<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::ijaa<DataVector, SpatialDim, Frame>&,
      const Scalar<DataVector>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&) =
      &two_index_constraint<SpatialDim, Frame, DataVector>;
  using base = TwoIndexConstraint<SpatialDim, Frame>;
};

/*!
 * \brief Compute item to get the three-index constraint for the
 * generalized harmonic evolution system.
 *
 * \details See `three_index_constraint()`. Can be retrieved using
 * `GeneralizedHarmonic::Tags::ThreeIndexConstraint`.
 */
template <size_t SpatialDim, typename Frame>
struct ThreeIndexConstraintCompute : ThreeIndexConstraint<SpatialDim, Frame>,
                                     db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::DerivSpacetimeMetric<SpatialDim, Frame>,
                 Phi<SpatialDim, Frame>>;
  static constexpr tnsr::iaa<DataVector, SpatialDim, Frame> (*function)(
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&) =
      &three_index_constraint<SpatialDim, Frame, DataVector>;
  using base = ThreeIndexConstraint<SpatialDim, Frame>;
};

/*!
 * \brief Compute item to get the four-index constraint for the
 * generalized harmonic evolution system.
 *
 * \details See `four_index_constraint()`. Can be retrieved using
 * `GeneralizedHarmonic::Tags::FourIndexConstraint`.
 */
template <size_t SpatialDim, typename Frame>
struct FourIndexConstraintCompute : FourIndexConstraint<SpatialDim, Frame>,
                                    db::ComputeTag {
  using argument_tags = tmpl::list<
      ::Tags::deriv<Phi<SpatialDim, Frame>, tmpl::size_t<SpatialDim>, Frame>>;
  static constexpr tnsr::iaa<DataVector, SpatialDim, Frame> (*function)(
      const tnsr::ijaa<DataVector, SpatialDim, Frame>&) =
      &four_index_constraint<SpatialDim, Frame, DataVector>;
  using base = FourIndexConstraint<SpatialDim, Frame>;
};

/*!
 * \brief Compute item to get combined energy in all constraints for the
 * generalized harmonic evolution system.
 *
 * \details See `constraint_energy()`. Can be retrieved using
 * `GeneralizedHarmonic::Tags::ConstraintEnergy`.
 */
template <size_t SpatialDim, typename Frame>
struct ConstraintEnergyCompute : ConstraintEnergy<SpatialDim, Frame>,
                                 db::ComputeTag {
  using argument_tags =
      tmpl::list<GaugeConstraint<SpatialDim, Frame>,
                 FConstraint<SpatialDim, Frame>,
                 TwoIndexConstraint<SpatialDim, Frame>,
                 ThreeIndexConstraint<SpatialDim, Frame>,
                 FourIndexConstraint<SpatialDim, Frame>,
                 gr::Tags::InverseSpatialMetric<SpatialDim, Frame, DataVector>,
                 gr::Tags::DetSpatialMetric<DataVector>>;
  static constexpr auto function(
      const tnsr::a<DataVector, SpatialDim, Frame>& gauge_constraint,
      const tnsr::a<DataVector, SpatialDim, Frame>& f_constraint,
      const tnsr::ia<DataVector, SpatialDim, Frame>& two_index_constraint,
      const tnsr::iaa<DataVector, SpatialDim, Frame>& three_index_constraint,
      const tnsr::iaa<DataVector, SpatialDim, Frame>& four_index_constraint,
      const tnsr::II<DataVector, SpatialDim, Frame>& inverse_spatial_metric,
      const Scalar<DataVector>& spatial_metric_determinant) noexcept {
    return constraint_energy<SpatialDim, Frame, DataVector>(
        gauge_constraint, f_constraint, two_index_constraint,
        three_index_constraint, four_index_constraint, inverse_spatial_metric,
        spatial_metric_determinant);
  }
  using base = ConstraintEnergy<SpatialDim, Frame>;
};
}  // namespace Tags
}  // namespace GeneralizedHarmonic
