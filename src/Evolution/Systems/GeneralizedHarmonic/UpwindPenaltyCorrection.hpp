// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace GeneralizedHarmonic {
/*!
 * \brief Computes the generalized harmonic upwind multipenalty boundary
 * correction.
 *
 * This implements the upwind multipenalty boundary correction term
 * \f$D_\beta\f$. We index the evolved variables using unhatted Greek letters,
 * and the characteristic variables using hatted Greek letters. The general form
 * of \f$D_\beta\f$ is given by:
 *
 * \f{align*}{
 *   \label{eq:pnpm upwind boundary term characteristics}
 *   D_\beta =
 *   T_{\beta\hat{\beta}}^{\mathrm{ext}}
 *   \Lambda^{\mathrm{ext},-}_{\hat{\beta}\hat{\alpha}}
 *   v^{\mathrm{ext}}_{\hat{\alpha}}
 *   -T_{\beta\hat{\beta}}^{\mathrm{int}}
 *   \Lambda^{\mathrm{int},-}_{\hat{\beta}\hat{\alpha}}
 *   v^{\mathrm{int}}_{\hat{\alpha}}.
 * \f}
 *
 * Note that Eq. (6.3) of \cite Teukolsky2015ega is not exactly what's
 * implemented since the boundary term given by Eq. (6.3) does not consistently
 * treat both sides of the interface on the same footing.
 *
 * For the first-order generalized harmonic system the correction is:
 *
 * \f{align}{
 *   \label{eq:pnpm upwind gh metric}
 *   D_{g_{ab}} &= \tilde{\lambda}_{v^{g}}^{\mathrm{ext}}
 *                v^{\mathrm{ext},g}_{ab}
 *                - \tilde{\lambda}_{v^{g}}^{\mathrm{int}}
 *                v^{\mathrm{int},g}_{ab}, \\
 *   \label{eq:pnpm upwind gh pi}
 *   D_{\Pi_{ab}}
 *              &= \frac{1}{2}\left(\tilde{\lambda}_{v^+}^{\mathrm{ext}}
 *                v^{\mathrm{ext},+}_{ab} +
 *                \tilde{\lambda}_{v^-}^{\mathrm{ext}}
 *                v^{\mathrm{ext},-}_{ab}\right)
 *                + \tilde{\lambda}_{v^g}^\mathrm{ext}\gamma_2
 *                v^{\mathrm{ext},g}_{ab}
 *                \notag \\
 *              &-\frac{1}{2}\left(\tilde{\lambda}_{v^+}^{\mathrm{int}}
 *                v^{\mathrm{int},+}_{ab} +
 *                \tilde{\lambda}_{v^-}^{\mathrm{int}}
 *                v^{\mathrm{int},-}_{ab}\right)
 *                - \tilde{\lambda}_{v^g}^\mathrm{int}\gamma_2
 *                v^{\mathrm{int},g}_{ab} , \\
 *   \label{eq:pnpm upwind gh phi}
 *   D_{\Phi_{iab}}
 *              &= \frac{1}{2}\left(\tilde{\lambda}_{v^+}^{\mathrm{ext}}
 *                v^{\mathrm{ext},+}_{ab}
 *                - \tilde{\lambda}_{v^-}^{\mathrm{ext}}
 *                v^{\mathrm{ext},-}_{ab}\right)n_i^{\mathrm{ext}}
 *                + \tilde{\lambda}_{v^0}^{\mathrm{ext}}
 *                v^{\mathrm{ext},0}_{iab}
 *                \notag \\
 *              &-
 *                \frac{1}{2}\left(\tilde{\lambda}_{v^+}^{\mathrm{int}}
 *                v^{\mathrm{int},+}_{ab}
 *                - \tilde{\lambda}_{v^-}^{\mathrm{int}}
 *                v^{\mathrm{int},-}_{ab}\right)n_i^{\mathrm{int}}
 *                - \tilde{\lambda}_{v^0}^{\mathrm{int}}
 *                v^{\mathrm{int},0}_{iab},
 * \f}
 *
 * with characteristic fields
 *
 * \f{align}{
 *   \label{eq:pnpm gh v g}
 *   v^{g}_{ab} &= g_{ab}, \\
 *   \label{eq:pnpm gh v 0}
 *   v^{0}_{iab} &= (\delta^k_i-n^k n_i)\Phi_{kab}, \\
 *   \label{eq:pnpm gh v pm}
 *   v^{\pm}_{ab} &= \Pi_{ab}\pm n^i\Phi_{iab} -\gamma_2 g_{ab},
 * \f}
 *
 * and characteristic speeds
 *
 * \f{align}{
 *   \lambda_{v^g} =& -(1+\gamma_1)\beta^i n_i -v^i_g n_i, \\
 *   \lambda_{v^0} =& -\beta^i n_i -v^i_g n_i, \\
 *   \lambda_{v^\pm} =& \pm \alpha - \beta^i n_i - v^i_g n_i,
 * \f}
 *
 * where \f$v_g^i\f$ is the mesh velocity. We have also defined
 *
 * \f{align}{
 *   \tilde{\lambda}_{\hat{\alpha}} =
 *    \left\{
 *        \begin{array}{ll}
 *          \lambda_{\hat{\alpha}} &
 *            \mathrm{if}\;\lambda_{\hat{\alpha}}\le 0.0 \\
 *          0 & \mathrm{otherwise}
 *        \end{array}\right.
 * \f}
 *
 * Note that we have assumed \f$n_i^{\mathrm{ext}}\f$ points in the same
 * direction as \f$n_i^{\mathrm{int}}\f$, but in the code we have them point in
 * opposite directions. If \f$n_i^{\mathrm{ext}}\f$ points in the opposite
 * direction the external speeds have their sign flipped and the \f$\pm\f$
 * fields and speeds reverse roles. Specifically, in the code we have:
 *
 * \f{align}{
 *   \label{eq:pnpm upwind gh metric code}
 *   D_{g_{ab}} &= \bar{\lambda}_{v^{g}}^{\mathrm{ext}}
 *                v^{\mathrm{ext},g}_{ab}
 *                - \tilde{\lambda}_{v^{g}}^{\mathrm{int}}
 *                v^{\mathrm{int},g}_{ab}, \\
 *   \label{eq:pnpm upwind gh pi code}
 *   D_{\Pi_{ab}}
 *              &= \frac{1}{2}\left(\bar{\lambda}_{v^+}^{\mathrm{ext}}
 *                v^{\mathrm{ext},+}_{ab} +
 *                \bar{\lambda}_{v^-}^{\mathrm{ext}}
 *                v^{\mathrm{ext},-}_{ab}\right)
 *                + \bar{\lambda}_{v^g}^\mathrm{ext}\gamma_2
 *                v^{\mathrm{ext},g}_{ab}
 *                \notag \\
 *              &-\frac{1}{2}\left(\tilde{\lambda}_{v^+}^{\mathrm{int}}
 *                v^{\mathrm{int},+}_{ab} +
 *                \tilde{\lambda}_{v^-}^{\mathrm{int}}
 *                v^{\mathrm{int},-}_{ab}\right)
 *                - \tilde{\lambda}_{v^g}^\mathrm{int}\gamma_2
 *                v^{\mathrm{int},g}_{ab} , \\
 *   \label{eq:pnpm upwind gh phi code}
 *   D_{\Phi_{iab}}
 *              &= \frac{1}{2}\left(\bar{\lambda}_{v^+}^{\mathrm{ext}}
 *                v^{\mathrm{ext},+}_{ab}
 *                - \bar{\lambda}_{v^-}^{\mathrm{ext}}
 *                v^{\mathrm{ext},-}_{ab}\right)n_i^{\mathrm{ext}}
 *                + \bar{\lambda}_{v^0}^{\mathrm{ext}}
 *                v^{\mathrm{ext},0}_{iab}
 *                \notag \\
 *              &-
 *                \frac{1}{2}\left(\tilde{\lambda}_{v^+}^{\mathrm{int}}
 *                v^{\mathrm{int},+}_{ab}
 *                - \tilde{\lambda}_{v^-}^{\mathrm{int}}
 *                v^{\mathrm{int},-}_{ab}\right)n_i^{\mathrm{int}}
 *                - \tilde{\lambda}_{v^0}^{\mathrm{int}}
 *                v^{\mathrm{int},0}_{iab},
 * \f}
 *
 * where
 *
 * \f{align}{
 *   \bar{\lambda}_{\hat{\alpha}} =
 *    \left\{
 *        \begin{array}{ll}
 *          -\lambda_{\hat{\alpha}} &
 *            \mathrm{if}\;-\lambda_{\hat{\alpha}}\le 0.0 \\
 *          0 & \mathrm{otherwise}
 *        \end{array}\right.
 * \f}
 */
template <size_t Dim>
struct UpwindPenaltyCorrection : tt::ConformsTo<dg::protocols::NumericalFlux> {
  struct CharSpeedNormalTimesVPlus : db::SimpleTag {
    using type = tnsr::iaa<DataVector, Dim, Frame::Inertial>;
  };
  struct CharSpeedNormalTimesVMinus : db::SimpleTag {
    using type = tnsr::iaa<DataVector, Dim, Frame::Inertial>;
  };
  struct CharSpeedGamma2VSpacetimeMetric : db::SimpleTag {
    using type = tnsr::aa<DataVector, Dim, Frame::Inertial>;
  };
  struct TensorCharSpeeds : db::SimpleTag {
    // use a tensor for sending the char speeds so they all go into a Variables
    // to reduce memory allocations.
    using type = tnsr::a<DataVector, 3, Frame::Inertial>;
  };

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Computes the upwind penalty flux for the first-order generalized "
      "harmonic system. It requires no options."};
  static std::string name() noexcept { return "UpwindPenalty"; }

  // clang-tidy: non-const reference
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT

  using variables_tags =
      tmpl::list<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>,
                 Tags::Pi<Dim, Frame::Inertial>,
                 Tags::Phi<Dim, Frame::Inertial>>;

  using package_field_tags =
      tmpl::list<Tags::VSpacetimeMetric<Dim, Frame::Inertial>,
                 Tags::VZero<Dim, Frame::Inertial>,
                 Tags::VPlus<Dim, Frame::Inertial>,
                 Tags::VMinus<Dim, Frame::Inertial>, CharSpeedNormalTimesVPlus,
                 CharSpeedNormalTimesVMinus, CharSpeedGamma2VSpacetimeMetric,
                 TensorCharSpeeds>;
  using package_extra_tags = tmpl::list<>;

  using argument_tags = tmpl::list<
      Tags::VSpacetimeMetric<Dim, Frame::Inertial>,
      Tags::VZero<Dim, Frame::Inertial>, Tags::VPlus<Dim, Frame::Inertial>,
      Tags::VMinus<Dim, Frame::Inertial>,
      Tags::CharacteristicSpeeds<Dim, Frame::Inertial>, Tags::ConstraintGamma2,
      ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  void package_data(
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          packaged_char_speed_v_spacetime_metric,
      gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
          packaged_char_speed_v_zero,
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          packaged_char_speed_v_plus,
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          packaged_char_speed_v_minus,
      gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
          packaged_char_speed_n_times_v_plus,
      gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
          packaged_char_speed_n_times_v_minus,
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          packaged_char_speed_gamma2_v_spacetime_metric,
      gsl::not_null<tnsr::a<DataVector, 3, Frame::Inertial>*>
          packaged_char_speeds,

      const tnsr::aa<DataVector, Dim, Frame::Inertial>& v_spacetime_metric,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& v_zero,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& v_plus,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& v_minus,
      const std::array<DataVector, 4>& char_speeds,
      const Scalar<DataVector>& constraint_gamma2,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
      const noexcept;

  void operator()(
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          spacetime_metric_boundary_correction,
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          pi_boundary_correction,
      gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
          phi_boundary_correction,

      const tnsr::aa<DataVector, Dim, Frame::Inertial>&
          char_speed_v_spacetime_metric_int,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& char_speed_v_zero_int,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& char_speed_v_plus_int,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& char_speed_v_minus_int,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>&
          char_speed_normal_times_v_plus_int,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>&
          char_speed_normal_times_v_minus_int,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>&
          char_speed_constraint_gamma2_v_spacetime_metric_int,
      const tnsr::a<DataVector, 3, Frame::Inertial>& char_speeds_int,

      const tnsr::aa<DataVector, Dim, Frame::Inertial>&
          char_speed_v_spacetime_metric_ext,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& char_speed_v_zero_ext,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& char_speed_v_plus_ext,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& char_speed_v_minus_ext,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>&
          char_speed_minus_normal_times_v_plus_ext,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>&
          char_speed_minus_normal_times_v_minus_ext,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>&
          char_speed_constraint_gamma2_v_spacetime_metric_ext,
      const tnsr::a<DataVector, 3, Frame::Inertial>& char_speeds_ext)
      const noexcept;
};
}  // namespace GeneralizedHarmonic
