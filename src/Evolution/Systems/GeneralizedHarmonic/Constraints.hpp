// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines functions to calculate the generalized harmonic constraints

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace GeneralizedHarmonic {
// @{
/*!
 * \brief Computes the generalized-harmonic 3-index constraint.
 *
 * \details Computes the generalized-harmonic 3-index constraint,
 * \f$C_{iab} = \partial_i\psi_{ab} - \Phi_{iab},\f$ which is
 * given by Eq. (26) of \cite Lindblom:2005qh
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::iaa<DataType, SpatialDim, Frame> three_index_constraint(
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void three_index_constraint(
    gsl::not_null<tnsr::iaa<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;
// @}

// @{
/*!
 * \brief Computes the generalized-harmonic gauge constraint.
 *
 * \details Computes the generalized-harmonic gauge constraint
 * [Eq. (40) of \cite Lindblom:2005qh],
 * \f[
 * C_a = H_a + g^{ij} \Phi_{ija} + t^b \Pi_{ba}
 * - \frac{1}{2} g^i_a \psi^{bc} \Phi_{ibc}
 * - \frac{1}{2} t_a \psi^{bc} \Pi_{bc},
 * \f]
 * where \f$H_a\f$ is the gauge function,
 * \f$\psi_{ab}\f$ is the spacetime metric,
 * \f$\Pi_{ab}=-t^c\partial_c \psi_{ab}\f$, and
 * \f$\Phi_{iab} = \partial_i\psi_{ab}\f$; \f$t^a\f$ is the timelike unit
 * normal vector to the spatial slice, \f$g^{ij}\f$ is the inverse spatial
 * metric, and \f$g^b_c = \delta^b_c + t^b t_c\f$.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> gauge_constraint(
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void gauge_constraint(
    gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;
// @}

// @{
/*!
 * \brief Computes the generalized-harmonic 2-index constraint.
 *
 * \details Computes the generalized-harmonic 2-index constraint
 * [Eq. (44) of \cite Lindblom:2005qh],
 * \f{eqnarray}{
 * C_{ia} &\equiv& g^{jk}\partial_j \Phi_{ika}
 * - \frac{1}{2} g_a^j\psi^{cd}\partial_j \Phi_{icd}
 * + t^b \partial_i \Pi_{ba}
 * - \frac{1}{2} t_a \psi^{cd}\partial_i\Pi_{cd}
 * \nonumber\\&&
 * + \partial_i H_a
 * + \frac{1}{2} g_a^j \Phi_{jcd} \Phi_{ief}
 * \psi^{ce}\psi^{df}
 * + \frac{1}{2} g^{jk} \Phi_{jcd} \Phi_{ike}
 * \psi^{cd}t^e t_a
 * \nonumber\\&&
 * - g^{jk}g^{mn}\Phi_{jma}\Phi_{ikn}
 * + \frac{1}{2} \Phi_{icd} \Pi_{be} t_a
 *                             \left(\psi^{cb}\psi^{de}
 *                       +\frac{1}{2}\psi^{be} t^c t^d\right)
 * \nonumber\\&&
 * - \Phi_{icd} \Pi_{ba} t^c \left(\psi^{bd}
 *                             +\frac{1}{2} t^b t^d\right)
 * + \frac{1}{2} \gamma_2 \left(t_a \psi^{cd}
 * - 2 \delta^c_a t^d\right) C_{icd}.
 * \f}
 * where \f$H_a\f$ is the gauge function,
 * \f$\psi_{ab}\f$ is the spacetime metric,
 * \f$\Pi_{ab}=-t^c\partial_c \psi_{ab}\f$, and
 * \f$\Phi_{iab} = \partial_i\psi_{ab}\f$; \f$t^a\f$ is the timelike unit
 * normal vector to the spatial slice, \f$g^{ij}\f$ is the inverse spatial
 * metric, and \f$g^b_c = \delta^b_c + t^b t_c\f$.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ia<DataType, SpatialDim, Frame> two_index_constraint(
    const tnsr::ia<DataType, SpatialDim, Frame>& d_gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const Scalar<DataType>& gamma2,
    const tnsr::iaa<DataType, SpatialDim, Frame>&
        three_index_constraint) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void two_index_constraint(
    gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::ia<DataType, SpatialDim, Frame>& d_gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const Scalar<DataType>& gamma2,
    const tnsr::iaa<DataType, SpatialDim, Frame>&
        three_index_constraint) noexcept;
// @}

// @{
/*!
 * \brief Computes the generalized-harmonic 4-index constraint.
 *
 * \details Computes the independent components of the generalized-harmonic
 * 4-index constraint. The constraint itself is given by Eq. (45) of
 * \cite Lindblom:2005qh,
 * \f{eqnarray}{
 * C_{ijab} = 2 \partial_{[i}\Phi_{j]ab},
 * \f}
 * where \f$\Phi_{iab} = \partial_i\psi_{ab}\f$. Because the constraint is
 * antisymmetric on the two spatial indices, here we compute and store
 * only the independent components of \f$C_{ijab}\f$. Specifically, we
 * compute
 * \f{eqnarray}{
 * D_{iab} \equiv \frac{1}{2} \epsilon_{i}{}^{jk} C_{jkab}
 * = \epsilon_{i}{}^{jk} \partial_j \Phi_{kab},
 * \f}
 * where \f$\epsilon_{ijk}\f$ is the flat-space Levi-Civita symbol,
 * which is raised and lowered with the Kronecker delta.
 * In terms
 * of \f$D_{iab}\f$, the full 4-index constraint is
 * \f{eqnarray}{
 * C_{jkab} = \epsilon^{i}{}_{jk} D_{iab}.
 * \f}
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::iaa<DataType, SpatialDim, Frame> four_index_constraint(
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void four_index_constraint(
    gsl::not_null<tnsr::iaa<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi) noexcept;
// @}

// @{
/*!
 * \brief Computes the generalized-harmonic F constraint.
 *
 * \details Computes the generalized-harmonic F constraint
 * [Eq. (43) of https://arXiv.org/abs/gr-qc/0512093v3],
 * \f{eqnarray}{
 * {\cal F}_a &\equiv&
 * \frac{1}{2} g_a^i \psi^{bc}\partial_i \Pi_{bc}
 * - g^{ij} \partial_i \Pi_{ja}
 * - g^{ij} t^b \partial_i \Phi_{jba}
 * + \frac{1}{2} t_a \psi^{bc} g^{ij} \partial_i \Phi_{jbc}
 * \nonumber \\ &&
 * + t_a g^{ij} \partial_i H_j
 * + g_a^i \Phi_{ijb} g^{jk}\Phi_{kcd} \psi^{bd} t^c
 * - \frac{1}{2} g_a^i \Phi_{ijb} g^{jk}
 *   \Phi_{kcd} \psi^{cd} t^b
 * \nonumber \\ &&
 * - g_a^i t^b \partial_i H_b
 * + g^{ij} \Phi_{icd} \Phi_{jba} \psi^{bc} t^d
 * - \frac{1}{2} t_a g^{ij} g^{mn} \Phi_{imc} \Phi_{njd}\psi^{cd}
 * \nonumber \\ &&
 * - \frac{1}{4}  t_a g^{ij}\Phi_{icd}\Phi_{jbe}
 *    \psi^{cb}\psi^{de}
 * + \frac{1}{4}  t_a \Pi_{cd} \Pi_{be}
 *    \psi^{cb}\psi^{de}
 * - g^{ij} H_i \Pi_{ja}
 * \nonumber \\ &&
 * - t^b g^{ij} \Pi_{b i} \Pi_{ja}
 * - \frac{1}{4}  g_a^i \Phi_{icd} t^c t^d \Pi_{be}
 *   \psi^{be}
 * + \frac{1}{2} t_a \Pi_{cd} \Pi_{be}\psi^{ce}
 *   t^d t^b
 * \nonumber \\ &&
 * + g_a^i \Phi_{icd} \Pi_{be} t^c t^b \psi^{de}
 * - g^{ij}\Phi_{iba} t^b \Pi_{je} t^e
 * - \frac{1}{2} g^{ij}\Phi_{icd} t^c t^d \Pi_{ja}
 * \nonumber \\ &&
 * - g^{ij} H_i \Phi_{jba} t^b
 * + g_{a}^i \Phi_{icd} H_b \psi^{bc} t^d
 * +\gamma_2\bigl(g^{id}{\cal C}_{ida}
 * -\frac{1}{2}  g_a^i\psi^{cd}{\cal C}_{icd}\bigr)
 * \nonumber \\ &&
 * + \frac{1}{2} t_a \Pi_{cd}\psi^{cd} H_b t^b
 * - t_a g^{ij} \Phi_{ijc} H_d \psi^{cd}
 * +\frac{1}{2}  t_a g^{ij} H_i \Phi_{jcd}\psi^{cd},
 * \f}
 * where \f$H_a\f$ is the gauge function,
 * \f$\psi_{ab}\f$ is the spacetime metric,
 * \f$\Pi_{ab}=-t^c\partial_c \psi_{ab}\f$, and
 * \f$\Phi_{iab} = \partial_i\psi_{ab}\f$; \f$t^a\f$ is the timelike unit
 * normal vector to the spatial slice, \f$g^{ij}\f$ is the inverse spatial
 * metric, and \f$g^b_c = \delta^b_c + t^b t_c\f$.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> f_constraint(
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::ia<DataType, SpatialDim, Frame>& d_gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const Scalar<DataType>& gamma2,
    const tnsr::iaa<DataType, SpatialDim, Frame>&
        three_index_constraint) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint(
    gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::ia<DataType, SpatialDim, Frame>& d_gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const Scalar<DataType>& gamma2,
    const tnsr::iaa<DataType, SpatialDim, Frame>&
        three_index_constraint) noexcept;
// @}
}  // namespace GeneralizedHarmonic
