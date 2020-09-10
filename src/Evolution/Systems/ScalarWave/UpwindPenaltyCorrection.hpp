// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "Options/Options.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename>
class Variables;

class DataVector;

namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

namespace Tags {
template <typename>
struct NormalDotFlux;
template <typename>
struct Normalized;
}  // namespace Tags

namespace PUP {
class er;
}  // namespace PUP
/// \endcond

// IWYU pragma: no_forward_declare Tensor

namespace ScalarWave {
/*!
 * \brief Computes the scalar wave upwind multipenalty boundary
 * correction.
 *
 * This implements the upwind multipenalty boundary correction term
 * \f$D_\alpha\f$. The general form is given by:
 *
 * \f{align*}{
 *   \label{eq:pnpm upwind boundary term characteristics}
 *   D_\beta =
 *   T_{\beta\hat{\beta}}^{\mathrm{ext}}
 *   \Lambda^{\mathrm{ext},-}_{\hat{\beta}\hat{\alpha}}
 *   v^{\mathrm{ext}}_{\hat{\alpha}}
 *   -T_{\beta\hat{\beta}}^{\mathrm{int}}
 *   \Lambda^{\mathrm{int},-}_{\hat{\beta}\hat{\alpha}}
 *   v^{\mathrm{int}}_{\hat{\alpha}},
 * \f}
 *
 * We denote the evolved fields by \f$u_{\alpha}\f$, the characteristic fields
 * by \f$v_{\hat{\alpha}}\f$, and implicitly sum over reapeated indices.
 * \f$T_{\alpha\hat{\alpha}}\f$ transforms characteristic fields to evolved
 * fields, while \f$\Lambda_{\hat{\alpha}\hat{\beta}}^-\f$ is a diagonal matrix
 * with only the negative characteristic speeds. The int and ext superscripts
 * denote quantities on the internal and external side of the mortar. Note that
 * Eq. (6.3) of \cite Teukolsky2015ega is not exactly what's implemented since
 * that boundary term does not consistently treat both sides of the interface on
 * the same footing.
 *
 * For the scalar wave system the correction is:
 *
 * \f{align}{
 *   D_{\Psi} &= \tilde{\lambda}_{v^{\Psi}}^{\mathrm{ext}}
 *                v^{\mathrm{ext},\Psi}
 *                - \tilde{\lambda}_{v^{\Psi}}^{\mathrm{int}}
 *                v^{\mathrm{int},\Psi}, \\
 *   D_{\Pi} &= \frac{1}{2}\left(\tilde{\lambda}_{v^+}^{\mathrm{ext}}
 *             v^{\mathrm{ext},+} +
 *             \tilde{\lambda}_{v^-}^{\mathrm{ext}}
 *             v^{\mathrm{ext},-}\right)
 *             + \tilde{\lambda}_{v^\Psi}^\mathrm{ext}\gamma_2
 *             v^{\mathrm{ext},\Psi}
 *             \notag \\
 *           &-\frac{1}{2}\left(\tilde{\lambda}_{v^+}^{\mathrm{int}}
 *             v^{\mathrm{int},+} +
 *             \tilde{\lambda}_{v^-}^{\mathrm{int}}
 *             v^{\mathrm{int},-}\right)
 *             - \tilde{\lambda}_{v^\Psi}^\mathrm{int}\gamma_2
 *             v^{\mathrm{int},\Psi} , \\
 *   D_{\Phi_{i}}
 *              &= \frac{1}{2}\left(\tilde{\lambda}_{v^+}^{\mathrm{ext}}
 *                v^{\mathrm{ext},+}
 *                - \tilde{\lambda}_{v^-}^{\mathrm{ext}}
 *                v^{\mathrm{ext},-}\right)n_i^{\mathrm{ext}}
 *                + \tilde{\lambda}_{v^0}^{\mathrm{ext}}
 *                v^{\mathrm{ext},0}_{i}
 *                \notag \\
 *              &-
 *                \frac{1}{2}\left(\tilde{\lambda}_{v^+}^{\mathrm{int}}
 *                v^{\mathrm{int},+}
 *                - \tilde{\lambda}_{v^-}^{\mathrm{int}}
 *                v^{\mathrm{int},-}\right)n_i^{\mathrm{int}}
 *                - \tilde{\lambda}_{v^0}^{\mathrm{int}}
 *                v^{\mathrm{int},0}_{i},
 * \f}
 *
 * with characteristic fields
 *
 * \f{align}{
 *   v^{\Psi} &= \Psi, \\
 *   v^{0}_{i} &= (\delta^k_i-n^k n_i)\Phi_{k}, \\
 *   v^{\pm} &= \Pi\pm n^i\Phi_{i} -\gamma_2 \Psi,
 * \f}
 *
 * and characteristic speeds
 *
 * \f{align}{
 *   \lambda_{v^\Psi} =& -v^i_g n_i, \\
 *   \lambda_{v^0} =& -v^i_g n_i, \\
 *   \lambda_{v^\pm} =& \pm 1 - v^i_g n_i,
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
 * direction as \f$n_i^{\mathrm{int}}\f$. If \f$n_i^{\mathrm{ext}}\f$ points in
 * the opposite direction the external speeds have their sign flipped and the
 * \f$\pm\f$ fields and their speeds reverse roles. Specifically, in the code we
 * have:
 *
 * \f{align}{
 *   D_{\Psi} &= \bar{\lambda}_{v^{\Psi}}^{\mathrm{ext}}
 *                v^{\mathrm{ext},\Psi}
 *                - \tilde{\lambda}_{v^{\Psi}}^{\mathrm{int}}
 *                v^{\mathrm{int},g}, \\
 *   D_{\Pi}
 *              &= \frac{1}{2}\left(\bar{\lambda}_{v^+}^{\mathrm{ext}}
 *                v^{\mathrm{ext},+} +
 *                \bar{\lambda}_{v^-}^{\mathrm{ext}}
 *                v^{\mathrm{ext},-}\right)
 *                + \bar{\lambda}_{v^\Psi}^\mathrm{ext}\gamma_2
 *                v^{\mathrm{ext},\Psi}
 *                \notag \\
 *              &-\frac{1}{2}\left(\tilde{\lambda}_{v^+}^{\mathrm{int}}
 *                v^{\mathrm{int},+} +
 *                \tilde{\lambda}_{v^-}^{\mathrm{int}}
 *                v^{\mathrm{int},-}\right)
 *                - \tilde{\lambda}_{v^\Psi}^\mathrm{int}\gamma_2
 *                v^{\mathrm{int},\Psi} , \\
 *   D_{\Phi_{i}}
 *              &= \frac{1}{2}\left(\bar{\lambda}_{v^+}^{\mathrm{ext}}
 *                v^{\mathrm{ext},+}
 *                - \bar{\lambda}_{v^-}^{\mathrm{ext}}
 *                v^{\mathrm{ext},-}\right)n_i^{\mathrm{ext}}
 *                + \bar{\lambda}_{v^0}^{\mathrm{ext}}
 *                v^{\mathrm{ext},0}_{i}
 *                \notag \\
 *              &-
 *                \frac{1}{2}\left(\tilde{\lambda}_{v^+}^{\mathrm{int}}
 *                v^{\mathrm{int},+}
 *                - \tilde{\lambda}_{v^-}^{\mathrm{int}}
 *                v^{\mathrm{int},-}\right)n_i^{\mathrm{int}}
 *                - \tilde{\lambda}_{v^0}^{\mathrm{int}}
 *                v^{\mathrm{int},0}_{i},
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
 private:
  struct NormalTimesVPlus : db::SimpleTag {
    using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
  };
  struct NormalTimesVMinus : db::SimpleTag {
    using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
  };
  struct Gamma2VPsi : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct CharSpeedsTensor : db::SimpleTag {
    using type = tnsr::a<DataVector, 3, Frame::Inertial>;
  };

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Computes the upwind penalty boundary correction for a scalar wave "
      "system. It requires no options."};
  static std::string name() noexcept { return "UpwindPenalty"; }

  // clang-tidy: non-const reference
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT

  using variables_tags = tmpl::list<Pi, Phi<Dim>, Psi>;

  using package_field_tags =
      tmpl::list<Tags::VPsi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus,
                 NormalTimesVPlus, NormalTimesVMinus, Gamma2VPsi,
                 CharSpeedsTensor>;
  using package_extra_tags = tmpl::list<>;

  using argument_tags =
      tmpl::list<Tags::VPsi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus,
                 Tags::CharacteristicSpeeds<Dim>, Tags::ConstraintGamma2,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  void package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_char_speed_v_psi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_char_speed_v_zero,
      gsl::not_null<Scalar<DataVector>*> packaged_char_speed_v_plus,
      gsl::not_null<Scalar<DataVector>*> packaged_char_speed_v_minus,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_char_speed_n_times_v_plus,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_char_speed_n_times_v_minus,
      gsl::not_null<Scalar<DataVector>*> packaged_char_speed_gamma2_v_psi,
      gsl::not_null<tnsr::a<DataVector, 3, Frame::Inertial>*>
          packaged_char_speeds,

      const Scalar<DataVector>& v_psi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
      const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
      const std::array<DataVector, 4>& char_speeds,
      const Scalar<DataVector>& constraint_gamma2,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
      const noexcept;

  void operator()(
      gsl::not_null<Scalar<DataVector>*> pi_boundary_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          phi_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> psi_boundary_correction,

      const Scalar<DataVector>& char_speed_v_psi_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& char_speed_v_zero_int,
      const Scalar<DataVector>& char_speed_v_plus_int,
      const Scalar<DataVector>& char_speed_v_minus_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          char_speed_normal_times_v_plus_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          char_speed_normal_times_v_minus_int,
      const Scalar<DataVector>& char_speed_constraint_gamma2_v_psi_int,
      const tnsr::a<DataVector, 3, Frame::Inertial>& char_speeds_int,

      const Scalar<DataVector>& char_speed_v_psi_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& char_speed_v_zero_ext,
      const Scalar<DataVector>& char_speed_v_plus_ext,
      const Scalar<DataVector>& char_speed_v_minus_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          char_speed_minus_normal_times_v_plus_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          char_speed_minus_normal_times_v_minus_ext,
      const Scalar<DataVector>& char_speed_constraint_gamma2_v_psi_ext,
      const tnsr::a<DataVector, 3, Frame::Inertial>& char_speeds_ext)
      const noexcept;
};
}  // namespace ScalarWave
