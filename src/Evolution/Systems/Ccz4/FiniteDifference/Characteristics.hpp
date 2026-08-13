// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"

/// \cond
class DataVector;
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
namespace Tags {
template <typename Tag>
struct Normalized;
}  // namespace Tags
/// \endcond

namespace Ccz4::fd {
static constexpr size_t Dim = System::volume_dim;

/// Helper function to compute the projector q_{ij} = gamma_{ij} - n_i n_j
template <typename DataType, typename Frame>
tnsr::ii<DataType, Dim, Frame> projector_dd(
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric,
    const tnsr::i<DataType, Dim, Frame>& unit_normal_one_form);

/// Helper function to compute the TT part of a symmetric rank-2 tensor
template <typename DataType, typename Frame>
tnsr::ii<DataType, Dim, Frame> compute_tt_symmetric_tensor(
    const tnsr::ii<DataType, Dim, Frame>& tensor,
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric,
    const tnsr::i<DataType, Dim, Frame>& unit_normal_one_form);

/// @{
/*!
 * \brief Compute the characteristic speeds for the SoCcz4 system.
 *
 * There are totally 23 characteristic fields, 2*2 from the tensor sector,
 * 2*5 from the vector sector, and 9 from the scalar sector. In the
 * following, we only compute 2+5+9=16 characteristic speeds as the
 * two transverse characteristics in the tensor and vector sectors
 * have the same speeds.
 *
 * We list the characteristic fields and speeds (superscripts) below.
 * See \ref characteristic_fields() for the definitions of the characteristic
 * fields.
 *
 * If \b Ccz4::fd::System::shifting_shift is true, the characteristic speeds
 * are:
 *
 * char_speeds[0]: $U_{ij}^{\alpha-\beta^n}$
 *
 * char_speeds[1]: $U_{ij}^{-\alpha-\beta^n}$
 *
 * char_speeds[2]: $U_i^{-\beta^n}$
 *
 * char_speeds[3]: $U_i^{\alpha-\beta^n}$
 *
 * char_speeds[4]: $U_i^{-\alpha-\beta^n}$
 *
 * char_speeds[5]: $U_i^{\lambda_+}$
 *
 * char_speeds[6]: $U_i^{\lambda_-}$
 *
 * char_speeds[7]: $U^{-\beta^n}$
 *
 * char_speeds[8]: $U_{(1)}^{\alpha-\beta^n}$
 *
 * char_speeds[9]: $U_{(1)}^{-\alpha-\beta^n}$
 *
 * char_speeds[10]: $U_{(2)}^{\alpha-\beta^n}$
 *
 * char_speeds[11]: $U_{(2)}^{-\alpha-\beta^n}$
 *
 * char_speeds[12]: $U^{\sqrt{2\alpha}-\beta^n}$
 *
 * char_speeds[13]: $U^{-\sqrt{2\alpha}-\beta^n}$
 *
 * char_speeds[14]: $U^{\mu_+}$
 *
 * char_speeds[15]: $U^{\mu_-}$
 *
 * where $\beta^n = \beta^i n_i$; $n_i$ is a spatial unit normal
 * (w.r.t. the physical background metric) one-form. And
 * $\mu_\pm = \pm v_s-\beta^n$,
 * $v_s = \frac{2\sqrt{f}}{\sqrt{3}\phi}$, and $\lambda_\pm = \pm\sqrt{f}/\phi -
 * \beta^n$.
 *
 * If \b Ccz4::fd::System::shifting_shift is false, the following characteristic
 * speeds are changed:
 *
 * char_speeds[2]: $U_i^0$
 *
 * char_speeds[5]: $U_i^{\lambda_+}$
 *
 * char_speeds[6]: $U_i^{\lambda_-}$
 *
 * char_speeds[7]: $U^0$
 *
 * char_speeds[14]: $U^{\mu_+}$
 *
 * char_speeds[15]: $U^{\mu_-}$
 *
 * where $\lambda_{\pm} = -\frac{1}{2}\beta^n \pm \frac{\sqrt{4f + (\beta^n
 * \phi)^2}}{2\phi}$ and $\mu_{\pm} = -\frac{1}{2}\beta^n \pm \frac{\sqrt{48f +
 * 9(\beta^n \phi)^2}}{6\phi}$.
 *
 * \note It is assumed that the algebraic constraints
 * $\text{det}(\tilde{\gamma}_{ij})=1$ and $\tilde{\gamma}^{ij}\tilde{A}_{ij}=0$
 * are enforced and lapse and shift are evolved. Otherwise, the system is
 * not strongly hyperbolic.
 *
 */
template <typename Frame>
std::array<DataVector, 16> characteristic_speeds(
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const Scalar<DataVector>& conformal_factor, double f,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form);

template <typename Frame>
void characteristic_speeds(
    gsl::not_null<std::array<DataVector, 16>*> char_speeds,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const Scalar<DataVector>& conformal_factor, double f,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form);

template <typename Frame>
struct CharacteristicSpeedsCompute : Tags::CharacteristicSpeeds<DataVector>,
                                     db::ComputeTag {
  using base = Tags::CharacteristicSpeeds<DataVector>;
  using type = typename base::type;
  using argument_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, Dim, Frame>,
      Ccz4::Tags::ConformalFactor<DataVector>,
      ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim, Frame>>>;
  using return_type = typename base::type;
  static void function(
      const gsl::not_null<return_type*> result, const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame>& shift,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form) {
    static constexpr double f = Ccz4::fd::System::f;
    characteristic_speeds(result, lapse, shift, conformal_factor, f,
                          unit_normal_one_form);
  };
};
/// @}

/// @{
/*!
 * \brief Compute the characteristic variables for the SoCcz4 system.
 *
 * Define the projector $q_{ij}=\gamma_{ij}-n_i n_j$ where $n_i$ is a spatial
 * unit normal (w.r.t. the physical background metric) one-form. Then for a
 * vector $v^i$ \f[ v_i^\perp := q_{ij}v^j, \qquad v_n := v^i n_i \f].
 *
 * If \b Ccz4::fd::System::shifting_shift is true, the characteristic fields
 * are:
 *
 * <b>%Tensor Sector</b>
 *
 * u_tnsr_plus/minus:
 * \f[
 * U^{\pm\alpha-\beta^n}_{ij}
 * ={\tilde A}^{TT}_{ij}\ \pm\ \frac12\,\partial_n{\tilde\gamma}^{TT}_{ij}.
 * \f]
 *
 * <b>%Vector Sector</b>
 *
 * u_vector1_zero:
 * \f[
 * U^{-\beta^n}_i= b^\perp_i-{\hat\Gamma}{}^\perp_i.
 * \f]
 * u_vector2_plus/minus:
 * \f[
 * U^{\pm\alpha-\beta^n}_i
 * ={\hat\Gamma}{}^\perp_i\
 * -\ \frac{1}{\phi^4}\partial_n{\tilde\gamma}^\perp_{ni}\
 * \mp\ \frac{2}{\phi^4}{\tilde A}^\perp_{ni}.
 * \f]
 * u_vector3_plus/minus:
 * \f[
 * U^{\lambda_\pm}_i
 * = b^\perp_i\ + V_\Gamma^\pm\hat{\Gamma}_i^\perp +
 * V_\beta^\pm\partial_n\beta^\perp_i. \f]
 * where $V_\Gamma^\pm = 0$ and
 * $V_\beta^\pm = \mp\ \frac{1}{\sqrt{f}\phi}$.
 *
 * <b>%Scalar Sector</b>
 *
 * u_scalar1_zero:
 * \f[
 * U^{-\beta^n}= b^{n}-{\hat\Gamma}{}^{\,n}.
 * \f]
 * u_scalar2_plus/minus:
 * \f[
 * U^{\pm\alpha-\beta^n}_{(1)}
 * ={\tilde
 * A}_{nn}\ \pm\ \frac12\,\partial_n{\tilde\gamma}_{nn}\
 * \pm\ 2\phi\,\partial_n\phi\
 * -\ \frac{2}{3}\phi^2 K. \f]
 * u_scalar3_plus/minus:
 * \f[
 * U^{\pm\alpha-\beta^n}_{(2)}
 * ={\hat\Gamma}{}^{\,n}\ +\ \frac{4}{\phi^3}\partial_n\phi\
 * \mp\ \frac{2}{\phi^2}\Theta.
 * \f]
 * u_scalar4_plus/minus:
 * \f[
 * U^{\pm\sqrt{2\alpha}-\beta^n}
 * =\partial_n\alpha\ \pm\ \sqrt{2\alpha}\,( K-2\Theta).
 * \f]
 * u_scalar5_plus/minus:
 * \f[
 * U^{\mu_\pm}
 * = b^{n}
 * + C_\phi^\pm\,\partial_n\phi + C_K^\pm K + C_\Theta^\pm \Theta + C_\Gamma^\pm
 * \hat{\Gamma}^{n}
 * + C_\alpha^\pm\,\partial_n\alpha + C_\beta^\pm\,\partial_n\beta^{n},
 * \f]
 * where
 * \f[ C_\phi^\pm = \frac{4\alpha^2}{-4f\phi+3\alpha^2\phi^3}, \qquad
 *     C_K^\pm = \pm \frac{4\alpha\sqrt f}{\sqrt3\,(2f\phi-3\alpha\phi^3)},
 * \qquad C_\Theta^\pm = \pm\frac{4\sqrt3\,\alpha\sqrt
 * f\,\big(-2f+\alpha(-1+2\alpha)\phi^2\big)}
 *          {\phi(-2f+3\alpha\phi^2)(-4f+3\alpha^2\phi^2)}, \f]
 * \f[ C_\Gamma^\pm = \frac{\alpha^2\phi^2}{-4f+3\alpha^2\phi^2}, \qquad
 *     C_\alpha^\pm = \frac{2\alpha}{2f-3\alpha\phi^2}, \qquad
 *     C_\beta^\pm = \mp\ \frac{2}{\sqrt3\,\sqrt f\,\phi}.\f]
 *
 * If \b Ccz4::fd::System::shifting_shift is false, the definitions of
 * u_vector3_plus/minus and u_scalar5_plus/minus are changed to:
 *
 * u_vector3_plus/minus:
 * \f[
 * U^{\lambda_\pm}_i
 * = b^\perp_i\ + V_\Gamma^\pm\hat{\Gamma}_i^\perp +
 * V_\beta^\pm\partial_n\beta^\perp_i. \f]
 * where $V_\Gamma^\pm=\beta^n\phi^2
 * V_\beta^\pm = \beta^n\phi^2 \left( \frac{\beta^n}{2f} \pm
 * \frac{\sqrt{4f+(\beta^n\phi)^2}}{2f\phi} \right)$
 *
 * u_scalar5_plus/minus:
 * \f[
 * U^{\mu_\pm}
 * = b^{n}
 * + C_\phi^\pm\,\partial_n\phi + C_K^\pm K + C_\Theta^\pm \Theta + C_\Gamma^\pm
 * \hat{\Gamma}^{n}
 * + C_\alpha^\pm\,\partial_n\alpha + C_\beta^\pm\,\partial_n\beta^{n},
 * \f]
 * where
 * \f[
 * C_\phi^\pm = -\frac{\alpha^2(\beta^n N^\pm + 8f)}{f\phi D_1^\pm}
 * \f]
 * \f[
 * C_K^\pm = -\frac{4\alpha N^\pm}{3\phi^2 D_2^\pm}
 * \f]
 * \f[
 * C_\Theta^\pm = \frac{2\alpha N^\pm(8f+\beta^n N^\mp\ +
 * 4(1-2\alpha)\alpha\phi^2)}{\phi^2 D_1^\pm D_2^\pm} \f]
 * \f[ C_\Gamma^\pm =
 * -\frac{\beta^n f N^\mp\ + \alpha^2(\beta^n N^\pm + 2f)\phi^2}{f D_1^\pm} \f]
 * \f[
 * C_\alpha^\pm = \frac{\alpha (N^\pm)^2}{6f\phi^2 D_2^\pm}
 * \f]
 * \f[
 * C_\beta^\pm = \frac{N^\pm}{6f\phi^2}
 * \f]
 * \f[
 * N^\pm = 3\beta^n\phi^2 \mp\ \phi\sqrt{48f+9(\beta^n\phi)^2}
 * \f]
 * \f[
 * D_1^\pm = 8f + \beta^n N^\mp\ - 6\alpha^2\phi^2
 * \f]
 * \f[
 * D_2^\pm = 8f + \beta^n N^\mp\ - 12\alpha\phi^2
 * \f]
 *
 * u_scalar1_zero and u_vector1_zero remain the same but with different
 * characteristic speeds.
 *
 * Note that the characteristic fields are defined up to transverse derivatives.
 * See \cite Gundlach:2005ta.
 *
 * \note It is assumed that the algebraic constraints
 * $\text{det}(\tilde{\gamma}_{ij})=1$ and $\tilde{\gamma}^{ij}\tilde{A}_{ij}=0$
 * are enforced and lapse and shift are evolved. Otherwise, the system is
 * not strongly hyperbolic.
 */
template <typename Frame>
typename Tags::CharacteristicFields<DataVector, Dim, Frame>::type
characteristic_fields(
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form,
    const tnsr::ii<DataVector, Dim, Frame>& conformal_spatial_metric,
    const Scalar<DataVector>& conformal_factor, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const tnsr::ii<DataVector, Dim, Frame>& a_tilde,
    const Scalar<DataVector>& theta,
    const tnsr::I<DataVector, Dim, Frame>& gamma_hat,
    const tnsr::I<DataVector, Dim, Frame>& auxiliary_field_b,
    const tnsr::ijj<DataVector, Dim, Frame>& d_conformal_spatial_metric,
    const tnsr::i<DataVector, Dim, Frame>& d_conformal_factor,
    const tnsr::i<DataVector, Dim, Frame>& d_lapse,
    const tnsr::iJ<DataVector, Dim, Frame>& d_shift, double f);

template <typename Frame>
void characteristic_fields(
    gsl::not_null<
        typename Tags::CharacteristicFields<DataVector, Dim, Frame>::type*>
        char_fields,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form,
    const tnsr::ii<DataVector, Dim, Frame>& conformal_spatial_metric,
    const Scalar<DataVector>& conformal_factor, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const tnsr::ii<DataVector, Dim, Frame>& a_tilde,
    const Scalar<DataVector>& theta,
    const tnsr::I<DataVector, Dim, Frame>& gamma_hat,
    const tnsr::I<DataVector, Dim, Frame>& auxiliary_field_b,
    const tnsr::ijj<DataVector, Dim, Frame>& d_conformal_spatial_metric,
    const tnsr::i<DataVector, Dim, Frame>& d_conformal_factor,
    const tnsr::i<DataVector, Dim, Frame>& d_lapse,
    const tnsr::iJ<DataVector, Dim, Frame>& d_shift, double f);

template <typename Frame>
struct CharacteristicFieldsCompute
    : Tags::CharacteristicFields<DataVector, Dim, Frame>,
      db::ComputeTag {
  using base = Tags::CharacteristicFields<DataVector, Dim, Frame>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<
      ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim, Frame>>,
      Ccz4::Tags::ConformalMetric<DataVector, Dim, Frame>,
      Ccz4::Tags::ConformalFactor<DataVector>, gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<DataVector, Dim, Frame>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      Ccz4::Tags::ATilde<DataVector, Dim, Frame>, Ccz4::Tags::Theta<DataVector>,
      Ccz4::Tags::GammaHat<DataVector, Dim, Frame>,
      Ccz4::Tags::AuxiliaryShiftB<DataVector, Dim, Frame>,
      ::Tags::deriv<Ccz4::Tags::ConformalMetric<DataVector, Dim, Frame>,
                    tmpl::size_t<Dim>, Frame>,
      ::Tags::deriv<Ccz4::Tags::ConformalFactor<DataVector>, tmpl::size_t<Dim>,
                    Frame>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>, Frame>,
      ::Tags::deriv<gr::Tags::Shift<DataVector, Dim, Frame>, tmpl::size_t<Dim>,
                    Frame>>;

  static void function(
      const gsl::not_null<return_type*> result,
      const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form,
      const tnsr::ii<DataVector, Dim, Frame>& conformal_spatial_metric,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame>& shift,
      const Scalar<DataVector>& trace_extrinsic_curvature,
      const tnsr::ii<DataVector, Dim, Frame>& a_tilde,
      const Scalar<DataVector>& theta,
      const tnsr::I<DataVector, Dim, Frame>& gamma_hat,
      const tnsr::I<DataVector, Dim, Frame>& auxiliary_field_b,
      const tnsr::ijj<DataVector, Dim, Frame>& d_conformal_spatial_metric,
      const tnsr::i<DataVector, Dim, Frame>& d_conformal_factor,
      const tnsr::i<DataVector, Dim, Frame>& d_lapse,
      const tnsr::iJ<DataVector, Dim, Frame>& d_shift) {
    static constexpr double f = Ccz4::fd::System::f;
    characteristic_fields(
        result, unit_normal_one_form, conformal_spatial_metric,
        conformal_factor, lapse, shift, trace_extrinsic_curvature, a_tilde,
        theta, gamma_hat, auxiliary_field_b, d_conformal_spatial_metric,
        d_conformal_factor, d_lapse, d_shift, f);
  };
};
/// @}

/// @{
/*!
 * \brief Compute the (normal derivatives of the) evolved fields from the
 * characteristic fields for the SoCcz4 system.
 *
 * Since the SoCcz4 system is second order in space, the isomorphism
 * between the characteristic space and the evolved space maps
 * characteristic fields to a mixture of evolved fields and their normal
 * derivatives. See \cite Gundlach:2005ta.
 *
 * For a characteristic field $U^\pm$, we define
 * \f[ \Sigma(U^\pm):=U^+ + U^-, \qquad \Delta(U^\pm):=U^+ - U^- \f].
 *
 * If \b Ccz4::fd::System::shifting_shift is true, the inverse characteristic
 * tansformation is given by
 * \f[
 * \partial_n\tilde{\gamma}^{TT}_{ij}
 * =
 * \Delta\!\left(U^{\pm\alpha-\beta^n}_{ij}\right),
 * \qquad
 * \tilde{A}^{TT}_{ij} =
 * \frac12\,\Sigma\!\left(U^{\pm\alpha-\beta^n}_{ij}\right).
 * \f]
 * \f[
 * \hat{\Gamma}_i^\perp
 * =
 * \frac{
 * \left(U_i^{\lambda_-}-U_i^{-\beta^n} \right)V_\beta^+
 * -
 * \left(U_i^{\lambda_+}-U_i^0 \right)V_\beta^-
 * }{V}
 * \f]
 * where
 * \f[
 * V=(1+V_\Gamma^-)V_\beta^+ - (1+V_\Gamma^+)V_\beta^-
 * \f]
 * \f[
 * \partial_n\beta_i^\perp
 * =
 * \frac{
 * (1+V_\Gamma^-)\left(U_i^{\lambda_+}-U_i^{-\beta^n}\right)
 * -
 * (1+V_\Gamma^+)\left(U_i^{\lambda_-}-U_i^{-\beta^n}\right)
 * }{V}
 * \f]
 * \f[
 * b_i^\perp = \hat{\Gamma}_i^\perp + U^{-\beta^n}_i,
 * \f]
 * \f[
 * \tilde{A}^\perp_{ni}
 * =
 * -\frac{\phi^4}{4}\,
 * \Delta\!\left(U^{\pm\alpha-\beta^n}_i\right).
 * \f]
 * \f[
 * \partial_n\tilde{\gamma}^\perp_{ni}
 * =
 * \phi^4\left(
 * \hat{\Gamma}_i^\perp
 * -
 * \frac12\,\Sigma\!\left(U^{\pm\alpha-\beta^n}_i\right)
 * \right).
 * \f]
 * \f[
 * \hat{\Gamma}^n=\frac{S^-C_\beta^+ - S^+C_\beta^-}{C}
 * \f]
 * where
 * \f[
 * \begin{aligned}
 * S^\pm=\ &U^{\mu_\pm}-U^0
 * -\frac{\phi^3}{8}C^\pm_\phi\Sigma\left(U_{(2)}^{\pm\alpha-\beta^n}\right)
 * - \frac{1}{2}C_\alpha^\pm\Sigma\left(U^{\pm\sqrt{2\alpha}-\beta^n}\right)\\
 * &- C_K^\pm\left[
 * \frac{1}{2\sqrt{2\alpha}}\Delta\left(U^{\pm\sqrt{2\alpha}-\beta^n}\right)
 * -\frac{\phi^2}{2}\Delta\left(U^{\pm\alpha-\beta^n}_{(2)}\right)\right]
 * + \frac{\phi^2}{4}C_\Theta^\pm\Delta\left(U^{\pm\alpha-\beta^n}_{(2)}\right)
 * \end{aligned}
 * \f]
 * \f[
 * C=\left(1-\frac{\phi^3}{4}C_\phi^-+C_\Gamma^-\right)C_\beta^+
 * -
 * \left(1-\frac{\phi^3}{4}C_\phi^++C_\Gamma^+\right)C_\beta^-
 * \f]
 *
 * \f[
 * \partial_n\beta^n
 * =
 * \frac{
 * \left(1-\frac{\phi^3}{4}C_\phi^-+C_\Gamma^-\right)S^+
 * -
 * \left(1-\frac{\phi^3}{4}C_\phi^++C_\Gamma^+\right)S^-
 * }{C}
 * \f]
 * \f[
 * \Theta \;=\;
 * -\frac{\phi^2}{4}\,\Delta\!\left(U_{(2)}^{\pm\alpha-\beta^n}\right). \f]
 * \f[
 * \partial_n\alpha
 * \;=\;\frac12\,\Sigma\!\left(U^{\pm\sqrt{2\alpha}-\beta^n}\right). \f]
 * \f[
 * K \;=\;
 * \frac{1}{2\sqrt{2\alpha}}\,\Delta\!\left(U^{\pm\sqrt{2\alpha}-\beta^n}\right)
 * \;-\;\frac{\phi^2}{2}\,\Delta\!\left(U_{(2)}^{\pm\alpha-\beta^n}\right).
 * \f]
 * \f[
 * b^n
 * =
 * U^{-\beta^n}
 * +
 * \hat{\Gamma}^n
 * \f]
 * \f[
 * \partial_n\phi
 * =
 * \frac{\phi^3}{8}\,\Sigma\!\left(U_{(2)}^{\pm\alpha-\beta^n}\right)
 * -
 * \frac{\phi^3}{4}\,
 * \hat{\Gamma}^n
 * \f]
 * \f[
 * \tilde A_{nn}
 * =
 * \frac12\,\Sigma\!\left(U_{(1)}^{\pm\alpha-\beta^n}\right)
 * +
 * \frac{2}{3}\phi^2\left[
 * \frac{1}{2\sqrt{2\alpha}}\,\Delta\!\left(U^{\pm\sqrt{2\alpha}-\beta^n}\right)
 * -\frac{\phi^2}{2}\,\Delta\!\left(U_{(2)}^{\pm\alpha-\beta^n}\right)
 * \right].
 * \f]
 * \f[
 * \partial_n\tilde\gamma_{nn}
 * =
 * \Delta\!\left(U_{(1)}^{\pm\alpha-\beta^n}\right)
 * -
 * \phi^4\left[
 * \frac{1}{2}\,\Sigma\!\left(U_{(2)}^{\pm\alpha-\beta^n}\right)
 * -
 * \hat{\Gamma}^n
 * \right].
 * \f]
 *
 * If \b Ccz4::fd::System::shifting_shift is false, the inverse transformation
 * is changed by replacing $U_i^{-\beta^n}$ and $U^{-\beta^n}$ above
 * with $U_i^0$ and $U^0$, respectively, and updating the coeffiecients
 * such as $V_\Gamma^\pm$, $V_\beta^\pm$, $C_\phi^\pm$, $C_K^\pm$, etc.
 *
 * \note It is assumed that the algebraic constraints
 * $\text{det}(\tilde{\gamma}_{ij})=1$ and $\tilde{\gamma}^{ij}\tilde{A}_{ij}=0$
 * are enforced and lapse and shift are evolved. Otherwise, the system is
 * not strongly hyperbolic.
 */
template <typename Frame>
typename Tags::EvolvedSpaceFromCharacteristicFields<DataVector, Dim,
                                                    Frame>::type
evolved_space_from_characteristic_fields(
    const tnsr::ii<DataVector, Dim, Frame>& u_tnsr_plus,
    const tnsr::ii<DataVector, Dim, Frame>& u_tnsr_minus,
    const tnsr::i<DataVector, Dim, Frame>& u_vector1_zero,
    const tnsr::i<DataVector, Dim, Frame>& u_vector2_plus,
    const tnsr::i<DataVector, Dim, Frame>& u_vector2_minus,
    const tnsr::i<DataVector, Dim, Frame>& u_vector3_plus,
    const tnsr::i<DataVector, Dim, Frame>& u_vector3_minus,
    const Scalar<DataVector>& u_scalar1_zero,
    const Scalar<DataVector>& u_scalar2_plus,
    const Scalar<DataVector>& u_scalar2_minus,
    const Scalar<DataVector>& u_scalar3_plus,
    const Scalar<DataVector>& u_scalar3_minus,
    const Scalar<DataVector>& u_scalar4_plus,
    const Scalar<DataVector>& u_scalar4_minus,
    const Scalar<DataVector>& u_scalar5_plus,
    const Scalar<DataVector>& u_scalar5_minus,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form,
    const tnsr::ii<DataVector, Dim, Frame>& conformal_spatial_metric,
    const Scalar<DataVector>& conformal_factor, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift, double f);

template <typename Frame>
void evolved_space_from_characteristic_fields(
    gsl::not_null<typename Tags::EvolvedSpaceFromCharacteristicFields<
        DataVector, Dim, Frame>::type*>
        evolved_space,
    const tnsr::ii<DataVector, Dim, Frame>& u_tnsr_plus,
    const tnsr::ii<DataVector, Dim, Frame>& u_tnsr_minus,
    const tnsr::i<DataVector, Dim, Frame>& u_vector1_zero,
    const tnsr::i<DataVector, Dim, Frame>& u_vector2_plus,
    const tnsr::i<DataVector, Dim, Frame>& u_vector2_minus,
    const tnsr::i<DataVector, Dim, Frame>& u_vector3_plus,
    const tnsr::i<DataVector, Dim, Frame>& u_vector3_minus,
    const Scalar<DataVector>& u_scalar1_zero,
    const Scalar<DataVector>& u_scalar2_plus,
    const Scalar<DataVector>& u_scalar2_minus,
    const Scalar<DataVector>& u_scalar3_plus,
    const Scalar<DataVector>& u_scalar3_minus,
    const Scalar<DataVector>& u_scalar4_plus,
    const Scalar<DataVector>& u_scalar4_minus,
    const Scalar<DataVector>& u_scalar5_plus,
    const Scalar<DataVector>& u_scalar5_minus,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form,
    const tnsr::ii<DataVector, Dim, Frame>& conformal_spatial_metric,
    const Scalar<DataVector>& conformal_factor, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift, double f);

template <typename Frame>
struct EvolvedSpaceFromCharacteristicFieldsCompute
    : Tags::EvolvedSpaceFromCharacteristicFields<DataVector, Dim, Frame>,
      db::ComputeTag {
  using base =
      Tags::EvolvedSpaceFromCharacteristicFields<DataVector, Dim, Frame>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<
      Tags::UTensorPlus<DataVector, Dim, Frame>,
      Tags::UTensorMinus<DataVector, Dim, Frame>,
      Tags::UVector1Zero<DataVector, Dim, Frame>,
      Tags::UVector2Plus<DataVector, Dim, Frame>,
      Tags::UVector2Minus<DataVector, Dim, Frame>,
      Tags::UVector3Plus<DataVector, Dim, Frame>,
      Tags::UVector3Minus<DataVector, Dim, Frame>,
      Tags::UScalar1Zero<DataVector>, Tags::UScalar2Plus<DataVector>,
      Tags::UScalar2Minus<DataVector>, Tags::UScalar3Plus<DataVector>,
      Tags::UScalar3Minus<DataVector>, Tags::UScalar4Plus<DataVector>,
      Tags::UScalar4Minus<DataVector>, Tags::UScalar5Plus<DataVector>,
      Tags::UScalar5Minus<DataVector>,
      ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim, Frame>>,
      Ccz4::Tags::ConformalMetric<DataVector, Dim, Frame>,
      Ccz4::Tags::ConformalFactor<DataVector>, gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<DataVector, Dim, Frame>>;

  static void function(
      const gsl::not_null<return_type*> evolved_space,
      const tnsr::ii<DataVector, Dim, Frame>& u_tnsr_plus,
      const tnsr::ii<DataVector, Dim, Frame>& u_tnsr_minus,
      const tnsr::i<DataVector, Dim, Frame>& u_vector1_zero,
      const tnsr::i<DataVector, Dim, Frame>& u_vector2_plus,
      const tnsr::i<DataVector, Dim, Frame>& u_vector2_minus,
      const tnsr::i<DataVector, Dim, Frame>& u_vector3_plus,
      const tnsr::i<DataVector, Dim, Frame>& u_vector3_minus,
      const Scalar<DataVector>& u_scalar1_zero,
      const Scalar<DataVector>& u_scalar2_plus,
      const Scalar<DataVector>& u_scalar2_minus,
      const Scalar<DataVector>& u_scalar3_plus,
      const Scalar<DataVector>& u_scalar3_minus,
      const Scalar<DataVector>& u_scalar4_plus,
      const Scalar<DataVector>& u_scalar4_minus,
      const Scalar<DataVector>& u_scalar5_plus,
      const Scalar<DataVector>& u_scalar5_minus,
      const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form,
      const tnsr::ii<DataVector, Dim, Frame>& conformal_spatial_metric,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame>& shift) {
    static constexpr double f = Ccz4::fd::System::f;
    evolved_space_from_characteristic_fields(
        evolved_space, u_tnsr_plus, u_tnsr_minus, u_vector1_zero,
        u_vector2_plus, u_vector2_minus, u_vector3_plus, u_vector3_minus,
        u_scalar1_zero, u_scalar2_plus, u_scalar2_minus, u_scalar3_plus,
        u_scalar3_minus, u_scalar4_plus, u_scalar4_minus, u_scalar5_plus,
        u_scalar5_minus, unit_normal_one_form, conformal_spatial_metric,
        conformal_factor, lapse, shift, f);
  };
};
/// @}
}  // namespace Ccz4::fd
