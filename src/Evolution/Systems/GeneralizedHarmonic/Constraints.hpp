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
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace GeneralizedHarmonic {
/// @{
/*!
 * \brief Computes the generalized-harmonic 3-index constraint.
 *
 * \details Computes the generalized-harmonic 3-index constraint,
 * \f$C_{iab} = \partial_i\psi_{ab} - \Phi_{iab},\f$ which is
 * given by Eq. (26) of \cite Lindblom2005qh
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::iaa<DataType, SpatialDim, Frame> three_index_constraint(
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi);

template <size_t SpatialDim, typename Frame, typename DataType>
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
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi);

template <size_t SpatialDim, typename Frame, typename DataType>
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

template <size_t SpatialDim, typename Frame, typename DataType>
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
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi);

template <size_t SpatialDim, typename Frame, typename DataType>
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
 * +\frac{1}{2}  t_a g^{ij} H_i \Phi_{jcd}\psi^{cd}
 * \nonumber \\ &&
 * - 16 \pi t^a T_{a b}
 * \f}
 * where \f$H_a\f$ is the gauge function,
 * \f$\psi_{ab}\f$ is the spacetime metric,
 * \f$\Pi_{ab}=-t^c\partial_c \psi_{ab}\f$, and
 * \f$\Phi_{iab} = \partial_i\psi_{ab}\f$; \f$t^a\f$ is the timelike unit
 * normal vector to the spatial slice, \f$g^{ij}\f$ is the inverse spatial
 * metric, \f$g^b_c = \delta^b_c + t^b t_c\f$, and \f$T_{a b}\f$ is the
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
 * {\cal C}_a = H_a + g^{i j} \Phi_{ij a} + t^b \Pi_{ba}
 * - \frac{1}{2} g_a{}^i \psi^{bc} \Phi_{i b c}
 * - \frac{1}{2} t_a \psi^{bc} \Pi_{b c}.
 * \f].
 *
 * Therefore, the Stress-energy contribution can be calculated from the
 * trace-reversed contribution appearing in
 * `grmhd::GhValenciaDivClean::add_stress_energy_term_to_dt_pi` -- the
 * trace reversal in that function and the trace-reversal that appears
 * explicitly in \f${\cal C}_a\f$ cancel.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
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

template <size_t SpatialDim, typename Frame, typename DataType>
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

template <size_t SpatialDim, typename Frame, typename DataType>
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

template <size_t SpatialDim, typename Frame, typename DataType>
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
     + C_{ia} C_{ja} g^{ij}\right)\nonumber\\
     & + & K_3 C_{iab} C_{jab} g^{ij} + K_4 C_{ikab} C_{jlab}g^{ij} g^{kl}.
 * \f}
 * Here \f$C_a\f$ is the gauge constraint, \f$F_a\f$ is the f constraint,
 * \f$C_{ia}\f$ is the two-index constraint, \f$C_{iab}\f$ is the
 * three-index constraint, \f$C_{ikab}\f$ is the four-index constraint,
 * \f$g^{ij}\f$ is the inverse spatial metric, and
 * \f$K_1\f$, \f$K_2\f$, \f$K_3\f$, and \f$K_4\f$ are constant multipliers
 * for each term that each default to a value of 1.0. Note that in this
 * equation, spacetime indices \f$a,b\f$ are raised and lowered with
 * the Kronecker delta.
 *
 * Also note that the argument `four_index_constraint` is a rank-3 tensor.
 * This is because `GeneralizedHarmonic::four_index_constraint()` takes
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
 *       + C_{ia} C_{ja} g^{ij}\right)\nonumber\\
 *   & + & K_3 C_{iab} C_{jab} g^{ij}\nonumber\\
     & + & 2 K_4 g D_{iab} D_{jab} g^{ij},
 * \f} where \f$g\f$ is the determinant of the spatial metric.
 *
 * To derive this expression for the constraint energy implemented here,
 * Eq.~(53) of \cite Lindblom2005qh is
 * \f{eqnarray}{
 * S_{AB} dc^A dc^B &=&
 *      m^{ab}\Bigl[d F_ad F_b
 *      +g^{ij}\bigl(d C_{ia}d C_{jb}
 *      +g^{kl}m^{cd}d C_{ikac}d C_{jlbd}\bigr)
 * \nonumber\\
 *      & + & \Lambda^2\bigl(d C_ad C_b
 *      +g^{ij}m^{cd}d C_{iac}d C_{jbd}\bigr)
 * \Bigr].
 * \f} Replace \f$dc^A\rightarrow c^A\f$ to get
 * \f{eqnarray}{
 * E&=&
 *      m^{ab}\Bigl[ F_a F_b
 *      +g^{ij}\bigl( C_{ia} C_{jb}
 *      +g^{kl}m^{cd} C_{ikac} C_{jlbd}\bigr)
 * \nonumber\\
 *      & + & \Lambda^2\bigl( C_a C_b
 *      +g^{ij}m^{cd} C_{iac} C_{jbd}\bigr)
 * \Bigr]\nonumber\\
 * &=&
 *      m^{ab} F_a F_b
 *      +m^{ab}g^{ij} C_{ia} C_{jb}
 *      +m^{ab}g^{ij} g^{kl}m^{cd} C_{ikac} C_{jlbd}
 * \nonumber\\
 *      & + & m^{ab}\Lambda^2 C_a C_b
 *      +m^{ab}\Lambda^2 g^{ij}m^{cd} C_{iac} C_{jbd}.
 * \f} Here \f$m^{ab}\f$ is an arbitrary positive-definite matrix, and
 * \f$\Lambda\f$ is an arbitrary real scalar.
 * Choose \f$m^{ab} = \delta^{ab}\f$ but allow an arbitrary coefficient to be
 * placed in front of each term. Then, absorb \f$\Lambda^2\f$ into one of
 * these coefficients, to get
 * \f{eqnarray}{ E &=& K_
 * F\delta^{ab} F_a F_b +K_2\delta^{ab}g^{ij} C_{ia} C_{jb}
 +K_4\delta^{ab}g^{ij}
 * g^{kl}\delta^{cd} C_{ikac} C_{jlbd}
 * \nonumber\\
 *      & + & K_1\delta^{ab} C_a C_b
 *      +K_3\delta^{ab} g^{ij}\delta^{cd} C_{iac} C_{jbd}.
 * \f}
 * Adopting a Euclidean norm for the constraint space (i.e., choosing to raise
 and
 * lower spacetime indices with Kronecker deltas) gives
 * \f{eqnarray}{ E &=& K_ F
 * F_a F_a +K_2g^{ij} C_{ia} C_{ja} +K_4 g^{ij} g^{kl} C_{ikac} C_{jlac}
 * \nonumber\\
 *      & + & K_1 C_a C_a
 *      +K_3g^{ij} C_{iac} C_{jac}.
 * \f} The two-index constraint and f constraint can be viewed as the
 * time and space components of a combined spacetime constraint. So next
 * choose
 * \f$K_ F=K_2\f$, giving \f{eqnarray}{ E&=& K_1 C_a C_a + K_2\left(F_a F_a
 *      + C_{ia} C_{ja} g^{ij}\right)\nonumber\\
 *      & + & K_3 C_{iab} C_{jab} g^{ij}
 *      + K_4 C_{ikab} C_{jlab}g^{ij} g^{kl}.
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
 *      + C_{ia} C_{ja} g^{ij}\right)\nonumber\\
 *      & + & K_3 C_{iab} C_{jab} g^{ij}\nonumber\\
 *      & + & K_4 D_{mab} D_{nab} \epsilon^{m}{}_{ik}
 *      \epsilon^{n}{}_{jl} g^{ij} g^{kl}. \f}
 *
 * There's a subtle point here: \f$g^{ij}\f$ is the inverse spatial metric,
 which
 * is not necessarily flat. But \f$\epsilon^{i}{}_{jk}\f$ is the flat space
 * Levi-Civita tensor. In order to raise and lower indices of the Levi-Civita
 * tensor with the inverse spatial metrics, put in the appropriate factors of
 * \f$\sqrt{g}\f$, where \f$g\f$ is the metric determinant, to make the
 * curved-space Levi-Civita tensor compatible with \f$g^{ij}\f$. Let
 * \f$\varepsilon^{ijk}\f$ represent the curved space Levi-Civita tensor
 compatible
 * with \f$g^{ij}\f$: \f{eqnarray}{
 * \varepsilon^{mik} = g^{-1/2} \epsilon^{mik}\\
 * \varepsilon_{mik} = g^{1/2} \epsilon_{mik}.
 * \f} Then we can write the constraint energy as
 * \f{eqnarray}{
 * E &=& K_1 C_a C_a + K_2\left(F_a F_a
 *      + C_{ia} C_{ja} g^{ij}\right)\nonumber\\
 *      & + & K_3 C_{iab} C_{jab} g^{ij}\nonumber\\
 *      & + & K_4 D_{mab} D_{nab} g g^{-1/2}\epsilon^{m}{}_{ik}
 * g^{-1/2}\epsilon^{n}{}_{jl} g^{ij} g^{kl}. \f} The factors of
 * \f$g^{-1/2}\f$ make the Levi-Civita tensor compatible with \f$g^{ij}\f$.
 * Swapping which summed indices are raised and which are lowered gives
 * \f{eqnarray}{ E &=& K_1 C_a
 C_a +
 * K_2\left(F_a F_a
 *      + C_{ia} C_{ja} g^{ij}\right)\nonumber\\
 *      & + & K_3 C_{iab} C_{jab} g^{ij}\nonumber\\
 *      & + & K_4 D_{mab} D_{nab} g g^{-1/2}\epsilon^{mik}
 g^{-1/2}\epsilon^{njl}
 * g_{ij} g_{kl}, \f} or \f{eqnarray}{ E &=& K_1 C_a C_a + K_2\left(F_a F_a
 *      + C_{ia} C_{ja} g^{ij}\right)\nonumber\\
 *      & + & K_3 C_{iab} C_{jab} g^{ij}\nonumber\\
 *      & + & K_4 D_{mab} D_{nab} g \varepsilon^{mik} \varepsilon^{njl} g_{ij}
 * g_{kl}, \f} or, reversing up and down repeated indices again,
 * \f{eqnarray}{ E
 * &=& K_1 C_a C_a + K_2\left(F_a F_a
 *      + C_{ia} C_{ja} g^{ij}\right)\nonumber\\
 *      & + & K_3 C_{iab} C_{jab} g^{ij}\nonumber\\
 *      & + & K_4 D_{mab} D_{nab} g \varepsilon^{m}{}_{ik}
 \varepsilon^{n}{}_{jl}
 * g^{ij} g^{kl}. \f}
 *
 * The metric raises and lowers the indices of \f$\varepsilon^{ijk}\f$,
 * so this can
 * be written as \f{eqnarray}{ E &=& K_1 C_a C_a + K_2\left(F_a F_a
 *      + C_{ia} C_{ja} g^{ij}\right)\nonumber\\
 *      & + & K_3 C_{iab} C_{jab} g^{ij}\nonumber\\
 *      & + & K_4 g D_{mab} D^{n}{}_{ab} \varepsilon^{mjl} \varepsilon_{njl}.
 * \f}
 *
 * Now, in flat space (Eq. (1.23) of \cite ThorneBlandford2017),
 * \f{eqnarray}{
 * \epsilon^{mjl} \epsilon_{njl} = \delta^{mj}_{nj} = \delta^m_n \delta^j_j -
 * \delta^m_j \delta^j_n = 2 \delta^m_n. \f} But this holds for curved space
 * as well: multiply the left hand side by \f$1 = g^{1/2} g^{-1/2}\f$ to get
 * \f{eqnarray}{
 * g^{-1/2}\epsilon^{mjl} g^{1/2}\epsilon_{njl} = \varepsilon^{mjl}
 * \varepsilon_{njl} = \delta^{mj}_{nj} = \delta^m_n \delta^j_j - \delta^m_j
 * \delta^j_n = 2 \delta^m_n. \f} So the constraint energy is \f{eqnarray}{ E
 &=&
 * K_1 C_a C_a + K_2\left(F_a F_a
 *      + C_{ia} C_{ja} g^{ij}\right)\nonumber\\
 *      & + & K_3 C_{iab} C_{jab} g^{ij}\nonumber\\
 *      & + & 2 K_4 D_{mab} D^{n}{}_{ab} \delta^m_n.
 * \f}
 * Simplifying gives the formula implemented here:
 * \f{eqnarray}{
 * E &=& K_1 C_a C_a + K_2\left(F_a F_a
 *      + C_{ia} C_{ja} g^{ij}\right)\nonumber\\
 *      & + & K_3 C_{iab} C_{jab} g^{ij}\nonumber\\
 *      & + & 2 K_4 g D_{iab} D_{jab} g^{ij}.
 * \f}
 */
template <size_t SpatialDim, typename Frame, typename DataType>
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

template <size_t SpatialDim, typename Frame, typename DataType>
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

namespace Tags {
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
      &gauge_constraint<SpatialDim, Frame, DataVector>);

  using base = GaugeConstraint<SpatialDim, Frame>;
};

/*!
 * \brief Compute item to get the F-constraint for the
 * generalized harmonic evolution system.
 *
 * \details See `f_constraint()`. Can be retrieved using
 * `GeneralizedHarmonic::Tags::FConstraint`.
 *
 * \note If the system contains matter, you will need to use a system-specific
 * version of this compute tag that passes the appropriate stress-energy tensor
 * to the F-constraint calculation.
 */
template <size_t SpatialDim, typename Frame>
struct FConstraintCompute : FConstraint<SpatialDim, Frame>, db::ComputeTag {
  using argument_tags = tmpl::list<
      GaugeH<SpatialDim, Frame>, SpacetimeDerivGaugeH<SpatialDim, Frame>,
      gr::Tags::SpacetimeNormalOneForm<SpatialDim, Frame, DataVector>,
      gr::Tags::SpacetimeNormalVector<SpatialDim, Frame, DataVector>,
      gr::Tags::InverseSpatialMetric<SpatialDim, Frame, DataVector>,
      gr::Tags::InverseSpacetimeMetric<SpatialDim, Frame, DataVector>,
      Pi<SpatialDim, Frame>, Phi<SpatialDim, Frame>,
      ::Tags::deriv<Pi<SpatialDim, Frame>, tmpl::size_t<SpatialDim>, Frame>,
      ::Tags::deriv<Phi<SpatialDim, Frame>, tmpl::size_t<SpatialDim>, Frame>,
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2,
      ThreeIndexConstraint<SpatialDim, Frame>>;

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
      &f_constraint<SpatialDim, Frame, DataVector>);

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
      SpacetimeDerivGaugeH<SpatialDim, Frame>,
      gr::Tags::SpacetimeNormalOneForm<SpatialDim, Frame, DataVector>,
      gr::Tags::SpacetimeNormalVector<SpatialDim, Frame, DataVector>,
      gr::Tags::InverseSpatialMetric<SpatialDim, Frame, DataVector>,
      gr::Tags::InverseSpacetimeMetric<SpatialDim, Frame, DataVector>,
      Pi<SpatialDim, Frame>, Phi<SpatialDim, Frame>,
      ::Tags::deriv<Pi<SpatialDim, Frame>, tmpl::size_t<SpatialDim>, Frame>,
      ::Tags::deriv<Phi<SpatialDim, Frame>, tmpl::size_t<SpatialDim>, Frame>,
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2,
      ThreeIndexConstraint<SpatialDim, Frame>>;

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
      &two_index_constraint<SpatialDim, Frame, DataVector>);

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
      tmpl::list<::Tags::deriv<gr::Tags::SpacetimeMetric<SpatialDim, Frame>,
                               tmpl::size_t<SpatialDim>, Frame>,
                 Phi<SpatialDim, Frame>>;

  using return_type = tnsr::iaa<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::iaa<DataVector, SpatialDim, Frame>*>,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&)>(
      &three_index_constraint<SpatialDim, Frame, DataVector>);

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

  using return_type = tnsr::iaa<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::iaa<DataVector, SpatialDim, Frame>*>,
      const tnsr::ijaa<DataVector, SpatialDim, Frame>&)>(
      &four_index_constraint<SpatialDim, Frame, DataVector>);

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
    destructive_resize_components(energy,
                                  get(spatial_metric_determinant).size());
    constraint_energy<SpatialDim, Frame, DataVector>(
        energy, gauge_constraint, f_constraint, two_index_constraint,
        three_index_constraint, four_index_constraint, inverse_spatial_metric,
        spatial_metric_determinant);
  }

  using base = ConstraintEnergy<SpatialDim, Frame>;
};
}  // namespace Tags
}  // namespace GeneralizedHarmonic
