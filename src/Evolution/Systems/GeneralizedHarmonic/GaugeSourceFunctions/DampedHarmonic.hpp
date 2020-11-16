// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace GeneralizedHarmonic {
namespace gauges {
/*!
 * \brief Damped harmonic gauge source function and its spacetime derivative.
 *
 * \details The gauge condition has been taken from \cite Szilagyi2009qz and
 * \cite Deppe2018uye. We provide both a "rollon" version
 * (`damped_harmonic_rollon`), and a "non-rollon" version (`damped_harmonic`).
 * In the non-rollon version the rollon function \f$R(t)=1\f$.
 *
 * \warning Only the non-rollon version can be used with a moving mesh.
 *
 * The covariant form of the source function \f$H_a\f$ is written as:
 *
 * \f{align*}
 * H_a :=  [1 - R(t)] H_a^\mathrm{init} +
 *  [\mu_{L1} \mathrm{log}(\sqrt{g}/N) + \mu_{L2} \mathrm{log}(1/N)] t_a
 *   - \mu_S g_{ai} N^i / N
 * \f}
 *
 * where \f$N, N^k\f$ are the lapse and shift respectively, \f$t_a\f$ is the
 * unit normal one-form to the spatial slice, and \f$g_{ab}\f$ is
 * the spatial metric (obtained by projecting the spacetime metric onto the
 * 3-slice, i.e. \f$g_{ab} = \psi_{ab} + t_a t_b\f$). The prefactors are:
 *
 * \f{align*}
 *  \mu_{L1} &= A_{L1} R(t) W(x^i) \mathrm{log}(\sqrt{g}/N)^{e_{L1}}, \\
 *  \mu_{L2} &= A_{L2} R(t) W(x^i) \mathrm{log}(1/N)^{e_{L2}}, \\
 *  \mu_{S} &= A_{S} R(t) W(x^i) \mathrm{log}(\sqrt{g}/N)^{e_{S}},
 * \f}
 *
 * temporal roll-on function \f$ R(t)\f$ is:
 *
 * \f{align*}
 * \begin{array}{ll}
 *     R(t) & = 0, & t< t_{0} \\
 *          & = 1 - \exp[-((t - t_{0})/ \sigma_t)^4], & t\geq t_{0} \\
 * \end{array}
 * \f}
 *
 * and the spatial weight function is:
 *
 * \f{align*}
 * W(x^i) = \exp[-(r/\sigma_r)^2].
 * \f}
 *
 * This weight function can be written with multiple constant factors in the
 * exponent in literature \cite Deppe2018uye, but we absorb them all into
 * \f$ \sigma_r\f$ here. The coordinate \f$ r\f$ is the Euclidean radius
 * in Inertial coordinates.
 *
 * Note that for the last three terms in \f$H_a\f$ (with \f$ X = \{L1, L2, S\}
 * \f$):
 *   - Amplitude factors \f$ A_{X} \f$ are taken as input here as `amp_coef_X`
 *   - Exponents \f$ e_X\f$ are taken as input here as `exp_X`.
 *   - Spatial weight function \f$W\f$ is specified completely by
 *     \f$\sigma_r\f$, which is taken as input here as `sigma_r`.
 *
 * Also computes spacetime derivatives, i.e. \f$\partial_a H_b\f$, of the damped
 * harmonic source function H. Using notation from damped_harmonic_h(), we
 * rewrite the same as:
 *
 * \f{align*}
 * \partial_a H_b =& \partial_a T_1 + \partial_a T_2 + \partial_a T_3, \\
 * H_a =& T_1 + T_2 + T_3,
 * \f}
 *
 * where:
 *
 * \f{align*}
 * T_1 =& [1 - R(t)] H_a^\mathrm{init}, \\
 * T_2 =& [\mu_{L1} \mathrm{log}(\sqrt{g}/N) + \mu_{L2} \mathrm{log}(1/N)] t_a,
 * \\
 * T_3 =& - \mu_S g_{ai} N^i / N.
 * \f}
 *
 * Derivation:
 *
 * \f$\blacksquare\f$ For \f$ T_1 \f$, the derivatives are:
 * \f{align*}
 * \partial_a T_1 = (1 - R(t))
 * \partial_a H_b^\mathrm{init}
 *                - H_b^\mathrm{init} \partial_a R.
 * \f}
 *
 * \f$\blacksquare\f$ Write \f$ T_2 \equiv (\mu_1 + \mu_2) t_b \f$. Then:
 *
 * \f{align*}
 * \partial_a T_2 =& (\partial_a \mu_1 + \partial_a \mu_2) t_b \\
 *               +& (\mu_1 + \mu_2) \partial_a t_b,
 * \f}
 *
 * where
 *
 * \f{align*}
 * \partial_a t_b =& \left(-\partial_a N, 0, 0, 0\right) \\
 *
 * \partial_a \mu_1
 *  =& \partial_a [A_{L1} R(t) W(x^i) \mathrm{log}(\sqrt{g}/N)^{e_{L1} +
 * 1}], \\
 *  =& A_{L1} R(t) W(x^i) \partial_a [\mathrm{log}(\sqrt{g}/N)^{e_{L1} +
 * 1}] \\
 *   +& A_{L1} \mathrm{log}(\sqrt{g}/N)^{e_{L1} + 1} \partial_a [R(t)
 * W(x^i)],\\
 *
 * \partial_a \mu_2
 *  =& \partial_a [A_{L2} R(t) W(x^i) \mathrm{log}(1/N)^{e_{L2} + 1}], \\
 *  =& A_{L2} R(t) W(x^i) \partial_a [\mathrm{log}(1/N)^{e_{L2} + 1}] \\
 *     +& A_{L2} \mathrm{log}(1/N)^{e_{L2} + 1} \partial_a [R(t) W(x^i)],
 * \f}
 *
 * where \f$\partial_a [R W] = \left(\partial_0 R(t), \partial_i
 * W(x^j)\right)\f$.
 *
 * \f$\blacksquare\f$ Finally, the derivatives of \f$ T_3 \f$ are:
 *
 * \f[
 * \partial_a T_3 = -\partial_a(\mu_S/N) g_{bi} N^i
 *                  -(\mu_S/N) \partial_a(g_{bi}) N^i
 *                  -(\mu_S/N) g_{bi}\partial_a N^i,
 * \f]
 *
 * where
 *
 * \f{align*}
 * \partial_a(\mu_S / N) =& (1/N)\partial_a \mu_S
 *                       - \frac{\mu_S}{N^2}\partial_a N, \,\,\mathrm{and}\\
 * \partial_a \mu_S =& \partial_a [A_S R(t) W(x^i)
 * \mathrm{log}(\sqrt{g}/N)^{e_S}], \\
 *                  =& A_S R(t) W(x^i) \partial_a
 * [\mathrm{log}(\sqrt{g}/N)^{e_S}] \\
 *                  +& A_S \mathrm{log}(\sqrt{g} / N)^{e_S} \partial_a [R(t)
 * W(x^i)].
 * \f}
 */
template <size_t SpatialDim, typename Frame>
void damped_harmonic_rollon(
    gsl::not_null<tnsr::a<DataVector, SpatialDim, Frame>*> gauge_h,
    gsl::not_null<tnsr::ab<DataVector, SpatialDim, Frame>*> d4_gauge_h,
    const tnsr::a<DataVector, SpatialDim, Frame>& gauge_h_init,
    const tnsr::ab<DataVector, SpatialDim, Frame>& dgauge_h_init,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame>& shift,
    const tnsr::a<DataVector, SpatialDim, Frame>&
    spacetime_unit_normal_one_form,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::II<DataVector, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataVector, SpatialDim, Frame>& phi, double time,
    const tnsr::I<DataVector, SpatialDim, Frame>& coords, double amp_coef_L1,
    double amp_coef_L2, double amp_coef_S, int exp_L1, int exp_L2, int exp_S,
    double rollon_start_time, double rollon_width, double sigma_r) noexcept;

/*!
 * \copydoc damped_harmonic_rollon()
 */
template <size_t SpatialDim, typename Frame>
void damped_harmonic(
    gsl::not_null<tnsr::a<DataVector, SpatialDim, Frame>*> gauge_h,
    gsl::not_null<tnsr::ab<DataVector, SpatialDim, Frame>*> d4_gauge_h,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame>& shift,
    const tnsr::a<DataVector, SpatialDim, Frame>&
        spacetime_unit_normal_one_form,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::II<DataVector, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataVector, SpatialDim, Frame>& phi,
    const tnsr::I<DataVector, SpatialDim, Frame>& coords, double amp_coef_L1,
    double amp_coef_L2, double amp_coef_S, int exp_L1, int exp_L2, int exp_S,
    double sigma_r) noexcept;
}  // namespace gauges
}  // namespace GeneralizedHarmonic
