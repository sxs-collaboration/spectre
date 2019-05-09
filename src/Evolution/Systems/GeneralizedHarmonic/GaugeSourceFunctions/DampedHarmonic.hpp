// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines Functions for calculating damped harmonic gauge quantities from
/// 3+1 quantities

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
class Time;
/// \endcond

// IWYU pragma: no_forward_declare Tags::Time
// IWYU pragma: no_forward_declare Tags::Coordinates
// IWYU pragma: no_forward_declare Tags::deriv
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare db::ComputeTag
// IWYU pragma: no_forward_declare gr::InverseSpatialMetric
// IWYU pragma: no_forward_declare gr::Tags::Lapse
// IWYU pragma: no_forward_declare gr::Tags::Shift
// IWYU pragma: no_forward_declare gr::Tags::SpacetimeMetric
// IWYU pragma: no_forward_declare gr::Tags::SpacetimeNormalOneForm
// IWYU pragma: no_forward_declare gr::Tags::SpacetimeNormalVector
// IWYU pragma: no_forward_declare gr::Tags::InverseSpatialMetric
// IWYU pragma: no_forward_declare gr::Tags::SqrtDetSpatialMetric

namespace GeneralizedHarmonic {
/*!
 * \brief Damped harmonic gauge source function.
 *
 * \details See damped_harmonic_h() for details.
 */
template <size_t SpatialDim, typename Frame>
struct DampedHarmonicHCompute : Tags::GaugeH<SpatialDim, Frame>,
                                db::ComputeTag {
  using argument_tags = tmpl::list<
      Tags::InitialGaugeH<SpatialDim, Frame>, ::gr::Tags::Lapse<DataVector>,
      ::gr::Tags::Shift<SpatialDim, Frame, DataVector>,
      ::gr::Tags::SqrtDetSpatialMetric<DataVector>,
      ::gr::Tags::SpacetimeMetric<SpatialDim, Frame, DataVector>, ::Tags::Time,
      OptionTags::GaugeHRollOnStartTime, OptionTags::GaugeHRollOnTimeWindow,
      ::Tags::Coordinates<SpatialDim, Frame>,
      OptionTags::GaugeHSpatialWeightDecayWidth<Frame>>;

  static typename db::item_type<Tags::GaugeH<SpatialDim, Frame>> function(
      const typename db::item_type<Tags::InitialGaugeH<SpatialDim, Frame>>&
          gauge_h_init,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, SpatialDim, Frame>& shift,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const tnsr::aa<DataVector, SpatialDim, Frame>& spacetime_metric,
      const Time& time, const double& t_start, const double& sigma_t,
      const tnsr::I<DataVector, SpatialDim, Frame>& coords,
      const double& sigma_r) noexcept;
};

/*!
 * \brief Damped harmonic gauge source function.
 *
 * \details Computes the gauge source function designed for binary black hole
 * evolutions. These have been taken from \cite Szilagyi2009qz and
 * \cite Deppe2018uye.
 *
 * The covariant form of the source function \f$H_a\f$ is written as:
 *
 * \f{align*}
 * H_a :=  [1 - R_{H_\mathrm{init}}(t)] H_a^\mathrm{init} +
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
 *  \mu_{L1} &= A_{L1} R_{L1}(t) W(x^i) \mathrm{log}(\sqrt{g}/N)^{e_{L1}}, \\
 *  \mu_{L2} &= A_{L2} R_{L2}(t) W(x^i) \mathrm{log}(1/N)^{e_{L2}}, \\
 *  \mu_{S} &= A_{S} R_{S}(t) W(x^i) \mathrm{log}(\sqrt{g}/N)^{e_{S}},
 * \f}
 *
 * temporal roll-on functions \f$ R_X(t)\f$ (with label \f$ X\f$) are:
 *
 * \f{align*}
 * \begin{array}{ll}
 *     R_X(t) & = 1 - \exp[-((t - t_{0,X})/ \sigma_t^X)^4], & t\geq t_{0,X} \\
 *     & = 0, & t< t_{0,X} \\
 * \end{array}
 * \f}
 *
 * and the spatial weight function is:
 *
 * \f{align*}
 * W(x^i) = \exp[-(r/\sigma_r)^2].
 * \f}
 * This weight function can be written with multiple constant factors in the
 * exponent in literature \cite Deppe2018uye, but we absorb them all into
 * \f$ \sigma_r\f$ here. The coordinate \f$ r\f$ is the Euclidean radius
 * in Inertial coordinates.
 *
 * Note:
 *   - Amplitude factors \f$ A_{X} \f$ are input as amp\_coef\_X
 *   - \f$ R_{X} \f$ is specified by \f$\{ t_{0,X}, \sigma_t^X\}\f$, which are
 *     input as \f$\{\f$t\_start\_X, sigma\_t\_X \f$\}\f$.
 *   - Exponents \f$ e_X\f$ are input as exp\_X.
 *   - The spatial weight function is specified completely by \f$\{\f$sigma\_r
 * \f$\}\f$.
 */
template <size_t SpatialDim, typename Frame>
void damped_harmonic_h(
    gsl::not_null<typename db::item_type<Tags::GaugeH<SpatialDim, Frame>>*>
        gauge_h,
    const typename db::item_type<Tags::InitialGaugeH<SpatialDim, Frame>>&
        gauge_h_init,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame>& shift,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& spacetime_metric,
    double time, const tnsr::I<DataVector, SpatialDim, Frame>& coords,
    // Scaling coeffs in front of each term
    double amp_coef_L1, double amp_coef_L2, double amp_coef_S,
    // exponents
    int exp_L1, int exp_L2, int exp_S,
    // roll on function parameters for lapse / shift terms
    double t_start_h_init, double sigma_t_h_init, double t_start_L1,
    double sigma_t_L1, double t_start_L2, double sigma_t_L2, double t_start_S,
    double sigma_t_S,
    // weight function
    double sigma_r) noexcept;

/*!
 * \brief Spacetime derivatives of the damped harmonic gauge source function.
 *
 * \details See spacetime_deriv_damped_harmonic_h() for details.
 */
template <size_t SpatialDim, typename Frame>
struct SpacetimeDerivDampedHarmonicHCompute
    : Tags::SpacetimeDerivGaugeH<SpatialDim, Frame>,
      db::ComputeTag {
  using argument_tags = tmpl::list<
      Tags::InitialGaugeH<SpatialDim, Frame>,
      Tags::SpacetimeDerivInitialGaugeH<SpatialDim, Frame>,
      ::gr::Tags::Lapse<DataVector>,
      ::gr::Tags::Shift<SpatialDim, Frame, DataVector>,
      ::gr::Tags::SpacetimeNormalOneForm<SpatialDim, Frame, DataVector>,
      ::gr::Tags::SqrtDetSpatialMetric<DataVector>,
      ::gr::Tags::InverseSpatialMetric<SpatialDim, Frame, DataVector>,
      ::gr::Tags::SpacetimeMetric<SpatialDim, Frame, DataVector>,
      Tags::Pi<SpatialDim, Frame>, Tags::Phi<SpatialDim, Frame>, ::Tags::Time,
      OptionTags::GaugeHRollOnStartTime, OptionTags::GaugeHRollOnTimeWindow,
      ::Tags::Coordinates<SpatialDim, Frame>,
      OptionTags::GaugeHSpatialWeightDecayWidth<Frame>>;

  static typename db::item_type<Tags::SpacetimeDerivGaugeH<SpatialDim, Frame>>
  function(
      const typename db::item_type<Tags::InitialGaugeH<SpatialDim, Frame>>&
          gauge_h_init,
      const typename db::item_type<
          Tags::SpacetimeDerivInitialGaugeH<SpatialDim, Frame>>& dgauge_h_init,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, SpatialDim, Frame>& shift,
      const tnsr::a<DataVector, SpatialDim, Frame>&
          spacetime_unit_normal_one_form,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const tnsr::II<DataVector, SpatialDim, Frame>& inverse_spatial_metric,
      const tnsr::aa<DataVector, SpatialDim, Frame>& spacetime_metric,
      const tnsr::aa<DataVector, SpatialDim, Frame>& pi,
      const tnsr::iaa<DataVector, SpatialDim, Frame>& phi, const Time& time,
      const double& t_start, const double& sigma_t,
      const tnsr::I<DataVector, SpatialDim, Frame>& coords,
      const double& sigma_r) noexcept;
};

/*!
 * \brief Spacetime derivatives of the damped harmonic gauge source function.
 *
 * \details Compute spacetime derivatives, i.e. \f$\partial_a H_b\f$, of the
 * damped harmonic source function H. Using notation from damped_harmonic_h(),
 * we rewrite the same as: \f{align*}
 * \partial_a H_b =& \partial_a T_1 + \partial_a T_2 + \partial_a T_3, \\
 * H_a =& T_1 + T_2 + T_3,
 * \f}
 *
 * where:
 *
 * \f{align*}
 * T_1 =& [1 - R_{H_\mathrm{init}}(t)] H_a^\mathrm{init}, \\
 * T_2 =& [\mu_{L1} \mathrm{log}(\sqrt{g}/N) + \mu_{L2} \mathrm{log}(1/N)] t_a,
 * \\ T_3 =& - \mu_S g_{ai} N^i / N. \f}
 *
 * Derivation:
 *
 * \f$\blacksquare\f$ For \f$ T_1 \f$, the derivatives are:
 * \f{align*}
 * \partial_a T_1 = (1 - R_{H_\mathrm{init}}(t))
 * \partial_a H_b^\mathrm{init}
 *                - H_b^\mathrm{init} \partial_a R_{H_\mathrm{init}}.
 * \f}
 *
 * \f$\blacksquare\f$ Write \f$ T_2 \equiv (\mu_1 + \mu_2) t_b \f$. Then:
 * \f{align*}
 * \partial_a T_2 =& (\partial_a \mu_1 + \partial_a \mu_2) t_b \\
 *               +& (\mu_1 + \mu_2) \partial_a t_b,
 * \f}
 * where
 * \f{align*}
 * \partial_a t_b =& \left(-\partial_a N, 0, 0, 0\right) \\
 *
 * \partial_a \mu_1
 *  =& \partial_a [A_{L1} R_{L1}(t) W(x^i) \mathrm{log}(\sqrt{g}/N)^{e_{L1} +
 * 1}], \\
 *  =& A_{L1} R_{L1}(t) W(x^i) \partial_a [\mathrm{log}(\sqrt{g}/N)^{e_{L1} +
 * 1}] \\
 *   +& A_{L1} \mathrm{log}(\sqrt{g}/N)^{e_{L1} + 1} \partial_a [R_{L1}(t)
 * W(x^i)],\\
 *
 * \partial_a \mu_2
 *  =& \partial_a [A_{L2} R_{L2}(t) W(x^i) \mathrm{log}(1/N)^{e_{L2} + 1}], \\
 *  =& A_{L2} R_{L2}(t) W(x^i) \partial_a [\mathrm{log}(1/N)^{e_{L2} + 1}] \\
 *     +& A_{L2} \mathrm{log}(1/N)^{e_{L2} + 1} \partial_a [R_{L2}(t) W(x^i)],
 * \f}
 * where \f$\partial_a [R W] = \left(\partial_0 R(t), \partial_i
 * W(x^j)\right)\f$.
 *
 * \f$\blacksquare\f$ Finally, the derivatives of \f$ T_3 \f$ are:
 * \f[
 * \partial_a T_3 = -\partial_a(\mu_S/N) g_{bi} N^i
 *                  -(\mu_S/N) \partial_a(g_{bi}) N^i
 *                  -(\mu_S/N) g_{bi}\partial_a N^i,
 * \f]
 * where
 * \f{align*}
 * \partial_a(\mu_S / N) =& (1/N)\partial_a \mu_S
 *                       - \frac{\mu_S}{N^2}\partial_a N, \,\,\mathrm{and}\\
 * \partial_a \mu_S =& \partial_a [A_S R_S(t) W(x^i)
 * \mathrm{log}(\sqrt{g}/N)^{e_S}], \\
 *                  =& A_S R_S(t) W(x^i) \partial_a
 * [\mathrm{log}(\sqrt{g}/N)^{e_S}] \\
 *                  +& A_S \mathrm{log}(\sqrt{g} / N)^{e_S} \partial_a [R_S(t)
 * W(x^i)]. \f}
 */
template <size_t SpatialDim, typename Frame>
void spacetime_deriv_damped_harmonic_h(
    gsl::not_null<
        typename db::item_type<Tags::SpacetimeDerivGaugeH<SpatialDim, Frame>>*>
        d4_gauge_h,
    const typename db::item_type<Tags::InitialGaugeH<SpatialDim, Frame>>&
        gauge_h_init,
    const typename db::item_type<
        Tags::SpacetimeDerivInitialGaugeH<SpatialDim, Frame>>& dgauge_h_init,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame>& shift,
    const tnsr::a<DataVector, SpatialDim, Frame>&
        spacetime_unit_normal_one_form,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::II<DataVector, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataVector, SpatialDim, Frame>& phi, double time,
    const tnsr::I<DataVector, SpatialDim, Frame>& coords,
    // Scaling coeffs in front of each term
    double amp_coef_L1, double amp_coef_L2, double amp_coef_S,
    // exponents
    int exp_L1, int exp_L2, int exp_S,
    // roll on function parameters for lapse / shift terms
    double t_start_h_init, double sigma_t_h_init, double t_start_L1,
    double sigma_t_L1, double t_start_L2, double sigma_t_L2, double t_start_S,
    double sigma_t_S,
    // weight function
    double sigma_r) noexcept;
}  // namespace GeneralizedHarmonic
