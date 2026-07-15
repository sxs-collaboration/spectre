// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines functions to calculate the generalized harmonic constraints

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintDampingTags.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace gh {
/// @{
/*!
 * \brief Computes the generalized-harmonic 3-index constraint.
 *
 * \details Computes the generalized-harmonic 3-index constraint,
 * \f$C_{iab} = \partial_i g_{ab} - \Phi_{iab},\f$ which is
 * given by Eq. (26) of \cite Lindblom2005qh
 */
template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::iaa<DataType, SpatialDim, Frame> three_index_constraint(
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi);

template <typename DataType, size_t SpatialDim, typename Frame>
void three_index_constraint(
    gsl::not_null<tnsr::iaa<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi);
/// @}

/// @{
/*!
 * \brief Computes the generalized-harmonic gauge constraint.
 *
 * \details Computes the generalized-harmonic gauge constraint
 * [Eq. (40) of \cite Lindblom2005qh],
 * \f[
 * C_a = H_a + \gamma^{ij} \Phi_{ija} + n^b \Pi_{ba}
 * - \frac{1}{2} \gamma^i_a g^{bc} \Phi_{ibc}
 * - \frac{1}{2} n_a g^{bc} \Pi_{bc},
 * \f]
 * where \f$H_a\f$ is the gauge function,
 * \f$g_{ab}\f$ is the spacetime metric,
 * \f$\Pi_{ab}=-n^c\partial_c g_{ab}\f$, and
 * \f$\Phi_{iab} = \partial_i g_{ab}\f$; \f$n^a\f$ is the timelike unit
 * normal vector to the spatial slice, \f$\gamma^{ij}\f$ is the inverse spatial
 * metric, and \f$\gamma^b_c = \delta^b_c + n^b n_c\f$.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::a<DataType, SpatialDim, Frame> gauge_constraint(
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi);

template <typename DataType, size_t SpatialDim, typename Frame>
void gauge_constraint(
    gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi);
/// @}

/// @{
/*!
 * \brief Computes the generalized-harmonic 2-index constraint.
 *
 * \details Computes the generalized-harmonic 2-index constraint
 * [Eq. (44) of \cite Lindblom2005qh],
 * \f{eqnarray}{
 * C_{ia} &\equiv& \gamma^{jk}\partial_j \Phi_{ika}
 * - \frac{1}{2} \gamma_a^j g^{cd}\partial_j \Phi_{icd}
 * + n^b \partial_i \Pi_{ba}
 * - \frac{1}{2} n_a g^{cd}\partial_i\Pi_{cd}
 * \nonumber\\&&
 * + \partial_i H_a
 * + \frac{1}{2} \gamma_a^j \Phi_{jcd} \Phi_{ief}
 * g^{ce}g^{df}
 * + \frac{1}{2} \gamma^{jk} \Phi_{jcd} \Phi_{ike}
 * g^{cd}n^e n_a
 * \nonumber\\&&
 * - \gamma^{jk}\gamma^{mn}\Phi_{jma}\Phi_{ikn}
 * + \frac{1}{2} \Phi_{icd} \Pi_{be} n_a
 *                             \left(g^{cb}g^{de}
 *                       +\frac{1}{2}g^{be} n^c n^d\right)
 * \nonumber\\&&
 * - \Phi_{icd} \Pi_{ba} n^c \left(g^{bd}
 *                             +\frac{1}{2} n^b n^d\right)
 * + \frac{1}{2} \gamma_2 \left(n_a g^{cd}
 * - 2 \delta^c_a n^d\right) C_{icd}.
 * \f}
 * where \f$H_a\f$ is the gauge function,
 * \f$g_{ab}\f$ is the spacetime metric,
 * \f$\Pi_{ab}=-n^c\partial_c g_{ab}\f$, and
 * \f$\Phi_{iab} = \partial_i g_{ab}\f$; \f$n^a\f$ is the timelike unit
 * normal vector to the spatial slice, \f$\gamma^{ij}\f$ is the inverse spatial
 * metric, and \f$\gamma^b_c = \delta^b_c + n^b n_c\f$.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::ia<DataType, SpatialDim, Frame> two_index_constraint(
    const tnsr::ab<DataType, SpatialDim, Frame>& spacetime_d_gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const Scalar<DataType>& gamma2,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint);

template <typename DataType, size_t SpatialDim, typename Frame>
void two_index_constraint(
    gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::ab<DataType, SpatialDim, Frame>& spacetime_d_gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const Scalar<DataType>& gamma2,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint);
/// @}

/// @{
/*!
 * \brief Computes the generalized-harmonic 4-index constraint.
 *
 * \details Computes the independent components of the generalized-harmonic
 * 4-index constraint. The constraint itself is given by Eq. (45) of
 * \cite Lindblom2005qh,
 * \f{eqnarray}{
 * C_{ijab} = 2 \partial_{[i}\Phi_{j]ab},
 * \f}
 * where \f$\Phi_{iab} = \partial_i g_{ab}\f$. Because the constraint is
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
template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::iaa<DataType, SpatialDim, Frame> four_index_constraint(
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi);

template <typename DataType, size_t SpatialDim, typename Frame>
void four_index_constraint(
    gsl::not_null<tnsr::iaa<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi);
/// @}

/// @{
/*!
 * \brief Computes the generalized-harmonic F constraint.
 *
 * \details Computes the generalized-harmonic F constraint
 * [Eq. (43) of \cite Lindblom2005qh],
 * \f{eqnarray}{
 * {\cal F}_a &\equiv&
 * \frac{1}{2} \gamma_a^i g^{bc}\partial_i \Pi_{bc}
 * - \gamma^{ij} \partial_i \Pi_{ja}
 * - \gamma^{ij} n^b \partial_i \Phi_{jba}
 * + \frac{1}{2} n_a g^{bc} \gamma^{ij} \partial_i \Phi_{jbc}
 * \nonumber \\ &&
 * + n_a \gamma^{ij} \partial_i H_j
 * + \gamma_a^i \Phi_{ijb} \gamma^{jk}\Phi_{kcd} g^{bd} n^c
 * - \frac{1}{2} \gamma_a^i \Phi_{ijb} \gamma^{jk}
 *   \Phi_{kcd} g^{cd} n^b
 * \nonumber \\ &&
 * - \gamma_a^i n^b \partial_i H_b
 * + \gamma^{ij} \Phi_{icd} \Phi_{jba} g^{bc} n^d
 * - \frac{1}{2} n_a \gamma^{ij} \gamma^{mn} \Phi_{imc} \Phi_{njd}g^{cd}
 * \nonumber \\ &&
 * - \frac{1}{4}  n_a \gamma^{ij}\Phi_{icd}\Phi_{jbe}
 *    g^{cb}g^{de}
 * + \frac{1}{4}  n_a \Pi_{cd} \Pi_{be}
 *    g^{cb}g^{de}
 * - \gamma^{ij} H_i \Pi_{ja}
 * \nonumber \\ &&
 * - n^b \gamma^{ij} \Pi_{b i} \Pi_{ja}
 * - \frac{1}{4}  \gamma_a^i \Phi_{icd} n^c n^d \Pi_{be}
 *   g^{be}
 * + \frac{1}{2} n_a \Pi_{cd} \Pi_{be}g^{ce}
 *   n^d n^b
 * \nonumber \\ &&
 * + \gamma_a^i \Phi_{icd} \Pi_{be} n^c n^b g^{de}
 * - \gamma^{ij}\Phi_{iba} n^b \Pi_{je} n^e
 * - \frac{1}{2} \gamma^{ij}\Phi_{icd} n^c n^d \Pi_{ja}
 * \nonumber \\ &&
 * - \gamma^{ij} H_i \Phi_{jba} n^b
 * + \gamma_{a}^i \Phi_{icd} H_b g^{bc} n^d
 * +\gamma_2\bigl(\gamma^{id}{\cal C}_{ida}
 * -\frac{1}{2}  \gamma_a^i g^{cd}{\cal C}_{icd}\bigr)
 * \nonumber \\ &&
 * + \frac{1}{2} n_a \Pi_{cd}g^{cd} H_b n^b
 * - n_a \gamma^{ij} \Phi_{ijc} H_d g^{cd}
 * +\frac{1}{2}  n_a \gamma^{ij} H_i \Phi_{jcd}g^{cd}
 * \nonumber \\ &&
 * - 16 \pi n^a T_{a b}
 * \f}
 * where \f$H_a\f$ is the gauge function,
 * \f$g_{ab}\f$ is the spacetime metric,
 * \f$\Pi_{ab}=-n^c\partial_c g_{ab}\f$, and
 * \f$\Phi_{iab} = \partial_i g_{ab}\f$; \f$n^a\f$ is the timelike unit
 * normal vector to the spatial slice, \f$\gamma^{ij}\f$ is the inverse spatial
 * metric, \f$\gamma^b_c = \delta^b_c + n^b n_c\f$, and \f$T_{a b}\f$ is the
 * stress-energy tensor if nonzero (if using the overload with no stress-energy
 * tensor provided, the stress energy term is omitted).
 *
 * To justify the stress-energy contribution to the F constraint, note that
 * the stress-energy tensor appears in the dynamics of the Generalized
 * Harmonic system only through \f$\partial_t \Pi_{a b}\f$.
 * That dependence arises from (using \f$\dots\f$ to indicate collections of
 * terms that are known to be independent of the stress-energy tensor):
 *
 * \f[
 * {\cal F}_a = \dots \alpha^{-1}(\partial_t {\cal C}_a),
 * \f]
 *
 * where
 *
 * \f[
 * {\cal C}_a = H_a + \gamma^{i j} \Phi_{ij a} + n^b \Pi_{ba}
 * - \frac{1}{2} \gamma_a{}^i g^{bc} \Phi_{i b c}
 * - \frac{1}{2} n_a g^{bc} \Pi_{b c}.
 * \f].
 *
 * Therefore, the Stress-energy contribution can be calculated from the
 * trace-reversed contribution appearing in
 * `grmhd::GhValenciaDivClean::add_stress_energy_term_to_dt_pi` -- the
 * trace reversal in that function and the trace-reversal that appears
 * explicitly in \f${\cal C}_a\f$ cancel.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::a<DataType, SpatialDim, Frame> f_constraint(
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::ab<DataType, SpatialDim, Frame>& spacetime_d_gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const Scalar<DataType>& gamma2,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint);

template <typename DataType, size_t SpatialDim, typename Frame>
void f_constraint(
    gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::ab<DataType, SpatialDim, Frame>& spacetime_d_gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const Scalar<DataType>& gamma2,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint);

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::a<DataType, SpatialDim, Frame> f_constraint(
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::ab<DataType, SpatialDim, Frame>& spacetime_d_gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const Scalar<DataType>& gamma2,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint,
    const tnsr::aa<DataType, SpatialDim, Frame>& trace_reversed_stress_energy);

template <typename DataType, size_t SpatialDim, typename Frame>
void f_constraint(
    gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::ab<DataType, SpatialDim, Frame>& spacetime_d_gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const Scalar<DataType>& gamma2,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint,
    const tnsr::aa<DataType, SpatialDim, Frame>& trace_reversed_stress_energy);
/// @}

/// @{
/*!
 * \brief Computes the generalized-harmonic (unnormalized) constraint energy.
 *
 * \details Computes the generalized-harmonic unnormalized constraint energy
 * [Eq. (53) of \cite Lindblom2005qh with \f$m^{ab}=\delta^{ab}\f$ and with each
 * term in the sum scaled by an arbitrary coefficient],
 * \f{eqnarray}{
 * E & = & K_1 C_a C_a + K_2\left(F_a F_a
     + C_{ia} C_{ja} \gamma^{ij}\right)\nonumber\\
     & + & K_3 C_{iab} C_{jab} \gamma^{ij} + K_4 C_{ikab} C_{jlab}\gamma^{ij}
 \gamma^{kl}.
 * \f}
 * Here \f$C_a\f$ is the gauge constraint, \f$F_a\f$ is the f constraint,
 * \f$C_{ia}\f$ is the two-index constraint, \f$C_{iab}\f$ is the
 * three-index constraint, \f$C_{ikab}\f$ is the four-index constraint,
 * \f$\gamma^{ij}\f$ is the inverse spatial metric, and
 * \f$K_1\f$, \f$K_2\f$, \f$K_3\f$, and \f$K_4\f$ are constant multipliers
 * for each term that each default to a value of 1.0. Note that in this
 * equation, spacetime indices \f$a,b\f$ are raised and lowered with
 * the Kronecker delta.
 *
 * Also note that the argument `four_index_constraint` is a rank-3 tensor.
 * This is because `gh::four_index_constraint()` takes
 * advantage of the antisymmetry of the four-index constraint's first two
 * indices to only compute and return the independent
 * components of \f$C_{ijab}\f$, which can be written as
 * \f{eqnarray}{
 * D_{iab} \equiv \frac{1}{2} \epsilon_{i}{}^{jk} C_{jkab},
 * \f} where \f$\epsilon_{ijk}\f$ is the flat-space Levi-Civita tensor,
 * whose inidces are raised and lowered with the Kronecker delta.
 * The result is
 * \f{eqnarray}{
 * E & = & K_1 C_a C_a + K_2\left(F_a F_a
 *       + C_{ia} C_{ja} \gamma^{ij}\right)\nonumber\\
 *   & + & K_3 C_{iab} C_{jab} \gamma^{ij}\nonumber\\
     & + & 2 K_4 \gamma D_{iab} D_{jab} \gamma^{ij},
 * \f} where \f$\gamma\f$ is the determinant of the spatial metric.
 *
 * To derive this expression for the constraint energy implemented here,
 * Eq.~(53) of \cite Lindblom2005qh is
 * \f{eqnarray}{
 * S_{AB} dc^A dc^B &=&
 *      m^{ab}\Bigl[d F_ad F_b
 *      +\gamma^{ij}\bigl(d C_{ia}d C_{jb}
 *      +\gamma^{kl}m^{cd}d C_{ikac}d C_{jlbd}\bigr)
 * \nonumber\\
 *      & + & \Lambda^2\bigl(d C_ad C_b
 *      +\gamma^{ij}m^{cd}d C_{iac}d C_{jbd}\bigr)
 * \Bigr].
 * \f} Replace \f$dc^A\rightarrow c^A\f$ to get
 * \f{eqnarray}{
 * E&=&
 *      m^{ab}\Bigl[ F_a F_b
 *      +\gamma^{ij}\bigl( C_{ia} C_{jb}
 *      +\gamma^{kl}m^{cd} C_{ikac} C_{jlbd}\bigr)
 * \nonumber\\
 *      & + & \Lambda^2\bigl( C_a C_b
 *      +\gamma^{ij}m^{cd} C_{iac} C_{jbd}\bigr)
 * \Bigr]\nonumber\\
 * &=&
 *      m^{ab} F_a F_b
 *      +m^{ab}\gamma^{ij} C_{ia} C_{jb}
 *      +m^{ab}\gamma^{ij} \gamma^{kl}m^{cd} C_{ikac} C_{jlbd}
 * \nonumber\\
 *      & + & m^{ab}\Lambda^2 C_a C_b
 *      +m^{ab}\Lambda^2 \gamma^{ij}m^{cd} C_{iac} C_{jbd}.
 * \f} Here \f$m^{ab}\f$ is an arbitrary positive-definite matrix, and
 * \f$\Lambda\f$ is an arbitrary real scalar.
 * Choose \f$m^{ab} = \delta^{ab}\f$ but allow an arbitrary coefficient to be
 * placed in front of each term. Then, absorb \f$\Lambda^2\f$ into one of
 * these coefficients, to get
 * \f{eqnarray}{ E &=& K_
 * F\delta^{ab} F_a F_b +K_2\delta^{ab}\gamma^{ij} C_{ia} C_{jb}
 +K_4\delta^{ab}\gamma^{ij}
 * \gamma^{kl}\delta^{cd} C_{ikac} C_{jlbd}
 * \nonumber\\
 *      & + & K_1\delta^{ab} C_a C_b
 *      +K_3\delta^{ab} \gamma^{ij}\delta^{cd} C_{iac} C_{jbd}.
 * \f}
 * Adopting a Euclidean norm for the constraint space (i.e., choosing to raise
 and
 * lower spacetime indices with Kronecker deltas) gives
 * \f{eqnarray}{ E &=& K_ F
 * F_a F_a +K_2\gamma^{ij} C_{ia} C_{ja} +K_4 \gamma^{ij} \gamma^{kl} C_{ikac}
 C_{jlac}
 * \nonumber\\
 *      & + & K_1 C_a C_a
 *      +K_3\gamma^{ij} C_{iac} C_{jac}.
 * \f} The two-index constraint and f constraint can be viewed as the
 * time and space components of a combined spacetime constraint. So next
 * choose
 * \f$K_ F=K_2\f$, giving \f{eqnarray}{ E&=& K_1 C_a C_a + K_2\left(F_a F_a
 *      + C_{ia} C_{ja} \gamma^{ij}\right)\nonumber\\
 *      & + & K_3 C_{iab} C_{jab} \gamma^{ij}
 *      + K_4 C_{ikab} C_{jlab}\gamma^{ij} \gamma^{kl}.
 * \f}
 *
 * Note that \f$C_{ikab}\f$ is antisymmetric on the first two indices. Next,
 * replace the four-index constraint \f$C_{ijab}\f$ with \f$D_{iab}\f$, which
 * contains the independent components of \f$C_{ijab}\f$. Specifically,
 * \f{eqnarray}{
 * D_{iab} \equiv \frac{1}{2} \epsilon_{i}{}^{jk} C_{jkab}.
 * \f} The inverse relationship is
 * \f{eqnarray}{
 * C_{jkab} = \epsilon^{i}{}_{jk} D_{iab},
 * \f} where \f$\epsilon_{ijk}\f$ is the flat-space Levi-Civita tensor, whose
 * indices are raised and lowered with the Kronecker delta. Inserting this
 relation
 * to replace \f$C_{jkab}\f$ with \f$D_{iab}\f$ gives \f{eqnarray}{ E &=& K_1
 C_a
 * C_a + K_2\left(F_a F_a
 *      + C_{ia} C_{ja} \gamma^{ij}\right)\nonumber\\
 *      & + & K_3 C_{iab} C_{jab} \gamma^{ij}\nonumber\\
 *      & + & K_4 D_{mab} D_{nab} \epsilon^{m}{}_{ik}
 *      \epsilon^{n}{}_{jl} \gamma^{ij} \gamma^{kl}. \f}
 *
 * There's a subtle point here: \f$\gamma^{ij}\f$ is the inverse spatial metric,
 which
 * is not necessarily flat. But \f$\epsilon^{i}{}_{jk}\f$ is the flat space
 * Levi-Civita tensor. In order to raise and lower indices of the Levi-Civita
 * tensor with the inverse spatial metrics, put in the appropriate factors of
 * \f$\sqrt{\gamma}\f$, where \f$\gamma\f$ is the metric determinant, to make
 the
 * curved-space Levi-Civita tensor compatible with \f$\gamma^{ij}\f$. Let
 * \f$\varepsilon^{ijk}\f$ represent the curved space Levi-Civita tensor
 compatible
 * with \f$\gamma^{ij}\f$: \f{eqnarray}{
 * \varepsilon^{mik} = \gamma^{-1/2} \epsilon^{mik}\\
 * \varepsilon_{mik} = \gamma^{1/2} \epsilon_{mik}.
 * \f} Then we can write the constraint energy as
 * \f{eqnarray}{
 * E &=& K_1 C_a C_a + K_2\left(F_a F_a
 *      + C_{ia} C_{ja} \gamma^{ij}\right)\nonumber\\
 *      & + & K_3 C_{iab} C_{jab} \gamma^{ij}\nonumber\\
 *      & + & K_4 D_{mab} D_{nab} \gamma \gamma^{-1/2}\epsilon^{m}{}_{ik}
 * \gamma^{-1/2}\epsilon^{n}{}_{jl} \gamma^{ij} \gamma^{kl}. \f} The factors of
 * \f$\gamma^{-1/2}\f$ make the Levi-Civita tensor compatible with
 \f$\gamma^{ij}\f$.
 * Swapping which summed indices are raised and which are lowered gives
 * \f{eqnarray}{ E &=& K_1 C_a
 C_a +
 * K_2\left(F_a F_a
 *      + C_{ia} C_{ja} \gamma^{ij}\right)\nonumber\\
 *      & + & K_3 C_{iab} C_{jab} \gamma^{ij}\nonumber\\
 *      & + & K_4 D_{mab} D_{nab} \gamma \gamma^{-1/2}\epsilon^{mik}
 \gamma^{-1/2}\epsilon^{njl}
 * \gamma_{ij} \gamma_{kl}, \f} or \f{eqnarray}{ E &=& K_1 C_a C_a +
 K_2\left(F_a F_a
 *      + C_{ia} C_{ja} \gamma^{ij}\right)\nonumber\\
 *      & + & K_3 C_{iab} C_{jab} \gamma^{ij}\nonumber\\
 *      & + & K_4 D_{mab} D_{nab} \gamma \varepsilon^{mik} \varepsilon^{njl}
 \gamma_{ij}
 * \gamma_{kl}, \f} or, reversing up and down repeated indices again,
 * \f{eqnarray}{ E
 * &=& K_1 C_a C_a + K_2\left(F_a F_a
 *      + C_{ia} C_{ja} \gamma^{ij}\right)\nonumber\\
 *      & + & K_3 C_{iab} C_{jab} \gamma^{ij}\nonumber\\
 *      & + & K_4 D_{mab} D_{nab} \gamma \varepsilon^{m}{}_{ik}
 \varepsilon^{n}{}_{jl}
 * \gamma^{ij} \gamma^{kl}. \f}
 *
 * The metric raises and lowers the indices of \f$\varepsilon^{ijk}\f$,
 * so this can
 * be written as \f{eqnarray}{ E &=& K_1 C_a C_a + K_2\left(F_a F_a
 *      + C_{ia} C_{ja} \gamma^{ij}\right)\nonumber\\
 *      & + & K_3 C_{iab} C_{jab} \gamma^{ij}\nonumber\\
 *      & + & K_4 \gamma D_{mab} D^{n}{}_{ab} \varepsilon^{mjl}
 \varepsilon_{njl}.
 * \f}
 *
 * Now, in flat space (Eq. (1.23) of \cite ThorneBlandford2017),
 * \f{eqnarray}{
 * \epsilon^{mjl} \epsilon_{njl} = \delta^{mj}_{nj} = \delta^m_n \delta^j_j -
 * \delta^m_j \delta^j_n = 2 \delta^m_n. \f} But this holds for curved space
 * as well: multiply the left hand side by \f$1 = \gamma^{1/2} \gamma^{-1/2}\f$
 to get
 * \f{eqnarray}{
 * \gamma^{-1/2}\epsilon^{mjl} \gamma^{1/2}\epsilon_{njl} = \varepsilon^{mjl}
 * \varepsilon_{njl} = \delta^{mj}_{nj} = \delta^m_n \delta^j_j - \delta^m_j
 * \delta^j_n = 2 \delta^m_n. \f} So the constraint energy is \f{eqnarray}{ E
 &=&
 * K_1 C_a C_a + K_2\left(F_a F_a
 *      + C_{ia} C_{ja} \gamma^{ij}\right)\nonumber\\
 *      & + & K_3 C_{iab} C_{jab} \gamma^{ij}\nonumber\\
 *      & + & 2 K_4 D_{mab} D^{n}{}_{ab} \delta^m_n.
 * \f}
 * Simplifying gives the formula implemented here:
 * \f{eqnarray}{
 * E &=& K_1 C_a C_a + K_2\left(F_a F_a
 *      + C_{ia} C_{ja} \gamma^{ij}\right)\nonumber\\
 *      & + & K_3 C_{iab} C_{jab} \gamma^{ij}\nonumber\\
 *      & + & 2 K_4 \gamma D_{iab} D_{jab} \gamma^{ij}.
 * \f}
 */
template <typename DataType, size_t SpatialDim, typename Frame>
Scalar<DataType> constraint_energy(
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& f_constraint,
    const tnsr::ia<DataType, SpatialDim, Frame>& two_index_constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& four_index_constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& spatial_metric_determinant,
    double gauge_constraint_multiplier = 1.0,
    double two_index_constraint_multiplier = 1.0,
    double three_index_constraint_multiplier = 1.0,
    double four_index_constraint_multiplier = 1.0);

template <typename DataType, size_t SpatialDim, typename Frame>
void constraint_energy(
    gsl::not_null<Scalar<DataType>*> energy,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& f_constraint,
    const tnsr::ia<DataType, SpatialDim, Frame>& two_index_constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& four_index_constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& spatial_metric_determinant,
    double gauge_constraint_multiplier = 1.0,
    double two_index_constraint_multiplier = 1.0,
    double three_index_constraint_multiplier = 1.0,
    double four_index_constraint_multiplier = 1.0);
/// @}

/// @{
/*!
 * \brief Computes the generalized-harmonic constraint energy normalization.
 *
 * \details Computes the generalized-harmonic normalized constraint energy
 * integrand [Eq. (70) of \cite Lindblom2005qh with \f$m^{ab}=\delta^{ab}\f$],
 * \f{eqnarray}{
 * ||E||=\gamma^{ij}\delta^{ab}\delta^{cd}
 * (\Lambda^{2}\partial_{i}g_{ac}\partial_{j}g_{bd}
 * + \partial_{i}\Pi_{ac}\partial_{j}\Pi_{bd} +
 * \gamma^{kl}\partial_{i}\Phi_{kac}\partial_{j}\Phi_{lbd})\gamma^{1/2}
 * \f}
 *
 * Here \f$\gamma^{ij}\f$ is the inverse spatial metric, and \f$\Lambda^{2}\f$
 * is the squared dimensional constant. Note
 * that in this equation, spacetime indices \f$a,b\f$ are raised and lowered
 * with the Kronecker delta.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
Scalar<DataType> constraint_energy_normalization(
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& sqrt_spatial_metric_determinant,
    double dimensional_constant);

template <typename DataType, size_t SpatialDim, typename Frame>
void constraint_energy_normalization(
    gsl::not_null<Scalar<DataType>*> energy_norm,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& sqrt_spatial_metric_determinant,
    double dimensional_constant);
/// @}

/// @{
/*!
 * \brief Computes the generalized-harmonic normalized constraint energy.
 *
 * \details Computes `constraint_energy()` and then normalizes by dividing
 * the unnormalized constraint energy. by `(eps +
 * constraint_energy_normalization())`, where eps (following SpEC) has a value
 * of 1.e-10, The `eps` term is present so that situations like Minkowski space,
 * in which `constraint_energy_normalization()` vanishes, do not cause
 * floating-point exceptions when computing the normalized constraint energy.
 */
template <typename DataType>
Scalar<DataType> normalized_constraint_energy(
    const Scalar<DataType>& constraint_energy,
    const Scalar<DataType>& constraint_energy_normalization);

template <typename DataType>
void normalized_constraint_energy(
    gsl::not_null<Scalar<DataType>*> normalized_energy,
    const Scalar<DataType>& constraint_energy,
    const Scalar<DataType>& constraint_energy_normalization);
/// @}

namespace Tags {
/*!
 * \brief Compute item to get the gauge constraint for the
 * generalized harmonic evolution system.
 *
 * \details See `gauge_constraint()`. Can be retrieved using
 * `gh::Tags::GaugeConstraint`.
 */
template <size_t SpatialDim, typename Frame>
struct GaugeConstraintCompute : GaugeConstraint<DataVector, SpatialDim, Frame>,
                                db::ComputeTag {
  using argument_tags = tmpl::list<
      GaugeH<DataVector, SpatialDim, Frame>,
      gr::Tags::SpacetimeNormalOneForm<DataVector, SpatialDim, Frame>,
      gr::Tags::SpacetimeNormalVector<DataVector, SpatialDim, Frame>,
      gr::Tags::InverseSpatialMetric<DataVector, SpatialDim, Frame>,
      gr::Tags::InverseSpacetimeMetric<DataVector, SpatialDim, Frame>,
      Pi<DataVector, SpatialDim, Frame>, Phi<DataVector, SpatialDim, Frame>>;

  using return_type = tnsr::a<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::a<DataVector, SpatialDim, Frame>*>,
      const tnsr::a<DataVector, SpatialDim, Frame>&,
      const tnsr::a<DataVector, SpatialDim, Frame>&,
      const tnsr::A<DataVector, SpatialDim, Frame>&,
      const tnsr::II<DataVector, SpatialDim, Frame>&,
      const tnsr::AA<DataVector, SpatialDim, Frame>&,
      const tnsr::aa<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&)>(
      &gauge_constraint<DataVector, SpatialDim, Frame>);

  using base = GaugeConstraint<DataVector, SpatialDim, Frame>;
};

/*!
 * \brief Compute item to get the F-constraint for the
 * generalized harmonic evolution system.
 *
 * \details See `f_constraint()`. Can be retrieved using
 * `gh::Tags::FConstraint`.
 *
 * \note If the system contains matter, you will need to use a system-specific
 * version of this compute tag that passes the appropriate stress-energy tensor
 * to the F-constraint calculation.
 */
template <size_t SpatialDim, typename Frame>
struct FConstraintCompute : FConstraint<DataVector, SpatialDim, Frame>,
                            db::ComputeTag {
  using argument_tags = tmpl::list<
      GaugeH<DataVector, SpatialDim, Frame>,
      SpacetimeDerivGaugeH<DataVector, SpatialDim, Frame>,
      gr::Tags::SpacetimeNormalOneForm<DataVector, SpatialDim, Frame>,
      gr::Tags::SpacetimeNormalVector<DataVector, SpatialDim, Frame>,
      gr::Tags::InverseSpatialMetric<DataVector, SpatialDim, Frame>,
      gr::Tags::InverseSpacetimeMetric<DataVector, SpatialDim, Frame>,
      Pi<DataVector, SpatialDim, Frame>, Phi<DataVector, SpatialDim, Frame>,
      ::Tags::deriv<Pi<DataVector, SpatialDim, Frame>, tmpl::size_t<SpatialDim>,
                    Frame>,
      ::Tags::deriv<Phi<DataVector, SpatialDim, Frame>,
                    tmpl::size_t<SpatialDim>, Frame>,
      ::gh::Tags::ConstraintGamma2,
      ThreeIndexConstraint<DataVector, SpatialDim, Frame>>;

  using return_type = tnsr::a<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::a<DataVector, SpatialDim, Frame>*>,
      const tnsr::a<DataVector, SpatialDim, Frame>&,
      const tnsr::ab<DataVector, SpatialDim, Frame>&,
      const tnsr::a<DataVector, SpatialDim, Frame>&,
      const tnsr::A<DataVector, SpatialDim, Frame>&,
      const tnsr::II<DataVector, SpatialDim, Frame>&,
      const tnsr::AA<DataVector, SpatialDim, Frame>&,
      const tnsr::aa<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::ijaa<DataVector, SpatialDim, Frame>&,
      const Scalar<DataVector>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&)>(
      &f_constraint<DataVector, SpatialDim, Frame>);

  using base = FConstraint<DataVector, SpatialDim, Frame>;
};

/*!
 * \brief Compute item to get the two-index constraint for the
 * generalized harmonic evolution system.
 *
 * \details See `two_index_constraint()`. Can be retrieved using
 * `gh::Tags::TwoIndexConstraint`.
 */
template <size_t SpatialDim, typename Frame>
struct TwoIndexConstraintCompute
    : TwoIndexConstraint<DataVector, SpatialDim, Frame>,
      db::ComputeTag {
  using argument_tags = tmpl::list<
      SpacetimeDerivGaugeH<DataVector, SpatialDim, Frame>,
      gr::Tags::SpacetimeNormalOneForm<DataVector, SpatialDim, Frame>,
      gr::Tags::SpacetimeNormalVector<DataVector, SpatialDim, Frame>,
      gr::Tags::InverseSpatialMetric<DataVector, SpatialDim, Frame>,
      gr::Tags::InverseSpacetimeMetric<DataVector, SpatialDim, Frame>,
      Pi<DataVector, SpatialDim, Frame>, Phi<DataVector, SpatialDim, Frame>,
      ::Tags::deriv<Pi<DataVector, SpatialDim, Frame>, tmpl::size_t<SpatialDim>,
                    Frame>,
      ::Tags::deriv<Phi<DataVector, SpatialDim, Frame>,
                    tmpl::size_t<SpatialDim>, Frame>,
      ::gh::Tags::ConstraintGamma2,
      ThreeIndexConstraint<DataVector, SpatialDim, Frame>>;

  using return_type = tnsr::ia<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::ia<DataVector, SpatialDim, Frame>*>,
      const tnsr::ab<DataVector, SpatialDim, Frame>&,
      const tnsr::a<DataVector, SpatialDim, Frame>&,
      const tnsr::A<DataVector, SpatialDim, Frame>&,
      const tnsr::II<DataVector, SpatialDim, Frame>&,
      const tnsr::AA<DataVector, SpatialDim, Frame>&,
      const tnsr::aa<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::ijaa<DataVector, SpatialDim, Frame>&,
      const Scalar<DataVector>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&)>(
      &two_index_constraint<DataVector, SpatialDim, Frame>);

  using base = TwoIndexConstraint<DataVector, SpatialDim, Frame>;
};

/*!
 * \brief Compute item to get the three-index constraint for the
 * generalized harmonic evolution system.
 *
 * \details See `three_index_constraint()`. Can be retrieved using
 * `gh::Tags::ThreeIndexConstraint`.
 */
template <size_t SpatialDim, typename Frame>
struct ThreeIndexConstraintCompute
    : ThreeIndexConstraint<DataVector, SpatialDim, Frame>,
      db::ComputeTag {
  using argument_tags = tmpl::list<
      ::Tags::deriv<gr::Tags::SpacetimeMetric<DataVector, SpatialDim, Frame>,
                    tmpl::size_t<SpatialDim>, Frame>,
      Phi<DataVector, SpatialDim, Frame>>;

  using return_type = tnsr::iaa<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::iaa<DataVector, SpatialDim, Frame>*>,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&)>(
      &three_index_constraint<DataVector, SpatialDim, Frame>);

  using base = ThreeIndexConstraint<DataVector, SpatialDim, Frame>;
};

/*!
 * \brief Compute item to get the four-index constraint for the
 * generalized harmonic evolution system.
 *
 * \details See `four_index_constraint()`. Can be retrieved using
 * `gh::Tags::FourIndexConstraint`.
 */
template <size_t SpatialDim, typename Frame>
struct FourIndexConstraintCompute
    : FourIndexConstraint<DataVector, SpatialDim, Frame>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<::Tags::deriv<Phi<DataVector, SpatialDim, Frame>,
                               tmpl::size_t<SpatialDim>, Frame>>;

  using return_type = tnsr::iaa<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::iaa<DataVector, SpatialDim, Frame>*>,
      const tnsr::ijaa<DataVector, SpatialDim, Frame>&)>(
      &four_index_constraint<DataVector, SpatialDim, Frame>);

  using base = FourIndexConstraint<DataVector, SpatialDim, Frame>;
};

/*!
 * \brief Compute item to get combined energy in all constraints for the
 * generalized harmonic evolution system.
 *
 * \details See `constraint_energy()`. Can be retrieved using
 * `gh::Tags::ConstraintEnergy`.
 */
template <size_t SpatialDim, typename Frame>
struct ConstraintEnergyCompute
    : ConstraintEnergy<DataVector, SpatialDim, Frame>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<GaugeConstraint<DataVector, SpatialDim, Frame>,
                 FConstraint<DataVector, SpatialDim, Frame>,
                 TwoIndexConstraint<DataVector, SpatialDim, Frame>,
                 ThreeIndexConstraint<DataVector, SpatialDim, Frame>,
                 FourIndexConstraint<DataVector, SpatialDim, Frame>,
                 gr::Tags::InverseSpatialMetric<DataVector, SpatialDim, Frame>,
                 gr::Tags::DetSpatialMetric<DataVector>>;

  using return_type = Scalar<DataVector>;

  static constexpr auto function(
      const gsl::not_null<Scalar<DataVector>*> energy,
      const tnsr::a<DataVector, SpatialDim, Frame>& gauge_constraint,
      const tnsr::a<DataVector, SpatialDim, Frame>& f_constraint,
      const tnsr::ia<DataVector, SpatialDim, Frame>& two_index_constraint,
      const tnsr::iaa<DataVector, SpatialDim, Frame>& three_index_constraint,
      const tnsr::iaa<DataVector, SpatialDim, Frame>& four_index_constraint,
      const tnsr::II<DataVector, SpatialDim, Frame>& inverse_spatial_metric,
      const Scalar<DataVector>& spatial_metric_determinant) {
    set_number_of_grid_points(energy, spatial_metric_determinant);
    constraint_energy<DataVector, SpatialDim, Frame>(
        energy, gauge_constraint, f_constraint, two_index_constraint,
        three_index_constraint, four_index_constraint, inverse_spatial_metric,
        spatial_metric_determinant);
  }

  using base = ConstraintEnergy<DataVector, SpatialDim, Frame>;
};

/*!
 * \brief Compute item to get the normalized combined energy in all constraints
 * for the generalized harmonic evolution system.
 *
 * \details See `normalized_constraint_energy()`. Can be retrieved using
 * `gh::Tags::NormalizedConstraintEnergy`.
 */
template <size_t SpatialDim, typename Frame>
struct NormalizedConstraintEnergyCompute
    : NormalizedConstraintEnergy<DataVector, SpatialDim, Frame>,
      db::ComputeTag {
  using argument_tags = tmpl::list<
      ConstraintEnergy<DataVector, SpatialDim, Frame>,
      ::Tags::deriv<gr::Tags::SpacetimeMetric<DataVector, SpatialDim, Frame>,
                    tmpl::size_t<SpatialDim>, Frame>,
      ::Tags::deriv<Pi<DataVector, SpatialDim, Frame>, tmpl::size_t<SpatialDim>,
                    Frame>,
      ::Tags::deriv<Phi<DataVector, SpatialDim, Frame>,
                    tmpl::size_t<SpatialDim>, Frame>,
      gr::Tags::InverseSpatialMetric<DataVector, SpatialDim, Frame>,
      gr::Tags::SqrtDetSpatialMetric<DataVector>>;

  using return_type = Scalar<DataVector>;

  static constexpr auto function(
      const gsl::not_null<Scalar<DataVector>*> normalized_energy,
      const Scalar<DataVector>& constraint_energy,
      const tnsr::iaa<DataVector, SpatialDim, Frame>& d_spacetime_metric,
      const tnsr::iaa<DataVector, SpatialDim, Frame>& d_pi,
      const tnsr::ijaa<DataVector, SpatialDim, Frame>& d_phi,
      const tnsr::II<DataVector, SpatialDim, Frame>& inverse_spatial_metric,
      const Scalar<DataVector>& sqrt_spatial_metric_determinant) {
    set_number_of_grid_points(normalized_energy, constraint_energy);
    const auto normalization =
        constraint_energy_normalization<DataVector, SpatialDim, Frame>(
            d_spacetime_metric, d_pi, d_phi, inverse_spatial_metric,
            sqrt_spatial_metric_determinant, 1.0);
    normalized_constraint_energy<DataVector>(normalized_energy,
                                             constraint_energy, normalization);
  }

  using base = NormalizedConstraintEnergy<DataVector, SpatialDim, Frame>;
};
}  // namespace Tags
}  // namespace gh
