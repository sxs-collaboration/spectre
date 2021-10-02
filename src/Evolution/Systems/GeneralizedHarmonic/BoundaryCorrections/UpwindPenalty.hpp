// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace GeneralizedHarmonic::BoundaryCorrections {
/*!
 * \brief Computes the generalized harmonic upwind multipenalty boundary
 * correction.
 *
 * This implements the upwind multipenalty boundary correction term
 * \f$D_\beta\f$. The general form is given by:
 *
 * \f{align*}{
 *   D_\beta =
 *   T_{\beta\hat{\beta}}^{\mathrm{ext}}
 *   \Lambda^{\mathrm{ext},-}_{\hat{\beta}\hat{\alpha}}
 *   v^{\mathrm{ext}}_{\hat{\alpha}}
 *   -T_{\beta\hat{\beta}}^{\mathrm{int}}
 *   \Lambda^{\mathrm{int},-}_{\hat{\beta}\hat{\alpha}}
 *   v^{\mathrm{int}}_{\hat{\alpha}}.
 * \f}
 *
 * We denote the evolved fields by \f$u_{\alpha}\f$, the characteristic fields
 * by \f$v_{\hat{\alpha}}\f$, and implicitly sum over reapeated indices.
 * \f$T_{\beta\hat{\beta}}\f$ transforms characteristic fields to evolved
 * fields, while \f$\Lambda_{\hat{\beta}\hat{\alpha}}^-\f$ is a diagonal matrix
 * with only the negative characteristic speeds, and has zeros on the diagonal
 * for all other entries. The int and ext superscripts denote quantities on the
 * internal and external side of the mortar. Note that Eq. (6.3) of
 * \cite Teukolsky2015ega is not exactly what's implemented since that boundary
 * term does not consistently treat both sides of the interface on the same
 * footing.
 *
 * For the first-order generalized harmonic system the correction is:
 *
 * \f{align*}{
 *   D_{g_{ab}} &= \lambda_{v^{g}}^{\mathrm{ext},-}
 *                v^{\mathrm{ext},g}_{ab}
 *                - \lambda_{v^{g}}^{\mathrm{int},-}
 *                v^{\mathrm{int},g}_{ab}, \\
 *   D_{\Pi_{ab}}
 *              &= \frac{1}{2}\left(\lambda_{v^+}^{\mathrm{ext},-}
 *                v^{\mathrm{ext},+}_{ab} +
 *                \lambda_{v^-}^{\mathrm{ext},-}
 *                v^{\mathrm{ext},-}_{ab}\right)
 *                + \lambda_{v^g}^{\mathrm{ext},-}\gamma_2
 *                v^{\mathrm{ext},g}_{ab}
 *                \notag \\
 *              &-\frac{1}{2}\left(\lambda_{v^+}^{\mathrm{int},-}
 *                v^{\mathrm{int},+}_{ab} +
 *                \lambda_{v^-}^{\mathrm{int},-}
 *                v^{\mathrm{int},-}_{ab}\right)
 *                - \lambda_{v^g}^{\mathrm{int},-}\gamma_2
 *                v^{\mathrm{int},g}_{ab} , \\
 *   D_{\Phi_{iab}}
 *              &= \frac{1}{2}\left(\lambda_{v^+}^{\mathrm{ext},-}
 *                v^{\mathrm{ext},+}_{ab}
 *                - \lambda_{v^-}^{\mathrm{ext},-}
 *                v^{\mathrm{ext},-}_{ab}\right)n_i^{\mathrm{ext}}
 *                + \lambda_{v^0}^{\mathrm{ext},-}
 *                v^{\mathrm{ext},0}_{iab}
 *                \notag \\
 *              &-
 *                \frac{1}{2}\left(\lambda_{v^+}^{\mathrm{int},-}
 *                v^{\mathrm{int},+}_{ab}
 *                - \lambda_{v^-}^{\mathrm{int},-}
 *                v^{\mathrm{int},-}_{ab}\right)n_i^{\mathrm{int}}
 *                - \lambda_{v^0}^{\mathrm{int},-}
 *                v^{\mathrm{int},0}_{iab},
 * \f}
 *
 * with characteristic fields
 *
 * \f{align*}{
 *   v^{g}_{ab} &= g_{ab}, \\
 *   v^{0}_{iab} &= (\delta^k_i-n^k n_i)\Phi_{kab}, \\
 *   v^{\pm}_{ab} &= \Pi_{ab}\pm n^i\Phi_{iab} -\gamma_2 g_{ab},
 * \f}
 *
 * and characteristic speeds
 *
 * \f{align*}{
 *   \lambda_{v^g} =& -(1+\gamma_1)\beta^i n_i -v^i_g n_i, \\
 *   \lambda_{v^0} =& -\beta^i n_i -v^i_g n_i, \\
 *   \lambda_{v^\pm} =& \pm \alpha - \beta^i n_i - v^i_g n_i,
 * \f}
 *
 * where \f$v_g^i\f$ is the mesh velocity and \f$n_i\f$ is the outward directed
 * unit normal covector to the interface. We have also defined
 *
 * \f{align}{
 *   \lambda^{\pm}_{\hat{\alpha}} =
 *    \left\{
 *        \begin{array}{ll}
 *          \lambda_{\hat{\alpha}} &
 *            \mathrm{if}\;\pm\lambda_{\hat{\alpha}}> 0 \\
 *          0 & \mathrm{otherwise}
 *        \end{array}\right.
 * \f}
 *
 * In the implementation we store the speeds in a rank-4 tensor with the zeroth
 * component being \f$\lambda_{v^\Psi}\f$, the first being \f$\lambda_{v^0}\f$,
 * the second being \f$\lambda_{v^+}\f$, and the third being
 * \f$\lambda_{v^-}\f$.
 *
 * Note that we have assumed \f$n_i^{\mathrm{ext}}\f$ points in the same
 * direction as \f$n_i^{\mathrm{int}}\f$, but in the code they point in opposite
 * directions. If \f$n_i^{\mathrm{ext}}\f$ points in the opposite direction the
 * external speeds have their sign flipped and the \f$\pm\f$ fields and their
 * speeds reverse roles (i.e. the \f$v^{\mathrm{ext},+}\f$ field is now flowing
 * into the element, while \f$v^{\mathrm{ext},-}\f$ flows out). In our
 * implementation this reversal actually cancels out, and we have the following
 * equations:
 *
 * \f{align*}{
 *   D_{g_{ab}} &= -\lambda_{v^{g}}^{\mathrm{ext},+}
 *                v^{\mathrm{ext},g}_{ab}
 *                - \lambda_{v^{g}}^{\mathrm{int},-}
 *                v^{\mathrm{int},g}_{ab}, \\
 *   D_{\Pi_{ab}}
 *              &= \frac{1}{2}\left(-\lambda_{v^+}^{\mathrm{ext},+}
 *                v^{\mathrm{ext},+}_{ab} -
 *                \lambda_{v^-}^{\mathrm{ext},+}
 *                v^{\mathrm{ext},-}_{ab}\right)
 *                - \lambda_{v^g}^{\mathrm{ext},+}\gamma_2
 *                v^{\mathrm{ext},g}_{ab}
 *                \notag \\
 *              &-\frac{1}{2}\left(\lambda_{v^+}^{\mathrm{int},-}
 *                v^{\mathrm{int},+}_{ab} +
 *                \lambda_{v^-}^{\mathrm{int},-}
 *                v^{\mathrm{int},-}_{ab}\right)
 *                - \lambda_{v^g}^{\mathrm{int},-}\gamma_2
 *                v^{\mathrm{int},g}_{ab} , \\
 *   D_{\Phi_{iab}}
 *              &= \frac{1}{2}\left(-\lambda_{v^+}^{\mathrm{ext},+}
 *                v^{\mathrm{ext},+}_{ab}
 *                + \lambda_{v^-}^{\mathrm{ext},+}
 *                v^{\mathrm{ext},-}_{ab}\right)n_i^{\mathrm{ext}}
 *                - \lambda_{v^0}^{\mathrm{ext},+}
 *                v^{\mathrm{ext},0}_{iab}
 *                \\
 *              &-
 *                \frac{1}{2}\left(\lambda_{v^+}^{\mathrm{int},-}
 *                v^{\mathrm{int},+}_{ab}
 *                - \lambda_{v^-}^{\mathrm{int},-}
 *                v^{\mathrm{int},-}_{ab}\right)n_i^{\mathrm{int}}
 *                - \lambda_{v^0}^{\mathrm{int},-}
 *                v^{\mathrm{int},0}_{iab},
 * \f}
 */
template <size_t Dim>
class UpwindPenalty final : public BoundaryCorrection<Dim> {
 private:
  struct NormalTimesVPlus : db::SimpleTag {
    using type = tnsr::iaa<DataVector, Dim, Frame::Inertial>;
  };
  struct NormalTimesVMinus : db::SimpleTag {
    using type = tnsr::iaa<DataVector, Dim, Frame::Inertial>;
  };
  struct Gamma2VSpacetimeMetric : db::SimpleTag {
    using type = tnsr::aa<DataVector, Dim, Frame::Inertial>;
  };
  struct CharSpeedsTensor : db::SimpleTag {
    using type = tnsr::a<DataVector, 3, Frame::Inertial>;
  };

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Computes the UpwindPenalty boundary correction term for the generalized "
      "harmonic system."};

  UpwindPenalty() = default;
  UpwindPenalty(const UpwindPenalty&) = default;
  UpwindPenalty& operator=(const UpwindPenalty&) = default;
  UpwindPenalty(UpwindPenalty&&) = default;
  UpwindPenalty& operator=(UpwindPenalty&&) = default;
  ~UpwindPenalty() override = default;

  /// \cond
  explicit UpwindPenalty(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(UpwindPenalty);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection<Dim>> get_clone() const override;

  using dg_package_field_tags =
      tmpl::list<Tags::VSpacetimeMetric<Dim, Frame::Inertial>,
                 Tags::VZero<Dim, Frame::Inertial>,
                 Tags::VPlus<Dim, Frame::Inertial>,
                 Tags::VMinus<Dim, Frame::Inertial>, NormalTimesVPlus,
                 NormalTimesVMinus, Gamma2VSpacetimeMetric, CharSpeedsTensor>;
  using dg_package_data_temporary_tags = tmpl::list<
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1,
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2,
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<Dim, Frame::Inertial, DataVector>>;
  using dg_package_data_primitive_tags = tmpl::list<>;
  using dg_package_data_volume_tags = tmpl::list<>;

  double dg_package_data(
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

      const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,

      const Scalar<DataVector>& constraint_gamma1,
      const Scalar<DataVector>& constraint_gamma2,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity) const;

  void dg_boundary_terms(
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          boundary_correction_spacetime_metric,
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          boundary_correction_pi,
      gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
          boundary_correction_phi,

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
          char_speed_normal_times_v_plus_ext,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>&
          char_speed_normal_times_v_minus_ext,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>&
          char_speed_constraint_gamma2_v_spacetime_metric_ext,
      const tnsr::a<DataVector, 3, Frame::Inertial>& char_speeds_ext,
      dg::Formulation /*dg_formulation*/) const;
};
}  // namespace GeneralizedHarmonic::BoundaryCorrections
