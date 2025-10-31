// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintDampingTags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
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

namespace gh::BoundaryCorrections {
/*!
 * \brief Computes the generalized harmonic upwind multipenalty boundary
 * correction using the averaged fields across the interface.
 *
 * The correction is given by
 *
 * \f{equation}{
 *   D_\beta =
 *   T^{-1}_{\beta\hat{\beta}}(u^{\text{avg}})
 *   \Lambda^-_{\hat{\beta}\hat{\alpha}}(u^{\text{avg}})
 *   T_{\hat{\alpha}\alpha}(u^{\text{avg}})
 *   \Delta u_\alpha,
 * \f}
 *
 * where $u_\alpha$ are the evolved fields, $\Lambda^-$ is the
 * diagonal matrix of incoming characteristic speeds,
 * $T_{\hat{\alpha}\alpha}$ transforms from evolved to characteristic
 * fields, and $\Delta u_\alpha = u_\alpha^{\text{ext}} -
 * u_\alpha^{\text{int}}$ is the discontinuity in the evolved fields.
 * All factors except the boundary discontinuity are calculated using
 * the averaged of the internal and external evolved fields:
 *
 * \f{equation}
 *   u^{\text{avg}} = \frac{u^{\text{int}} + u^{\text{ext}}}{2}.
 * \f}
 *
 * For the first-order generalized harmonic system the correction is:
 *
 * \f{align}{
 *   D_{g_{ab}} &= \lambda_{v^g}^- \Delta g_{ab} \\
 *   D_{\Pi_{ab}}
 *     &= \left(\lambda_{v^g}^-
 *              - \frac{\lambda_{v^+}^- + \lambda_{v^-}^-}{2}\right)
 *       \gamma_2 \Delta g_{ab}
 *     + \frac{\lambda_{v^+}^- + \lambda_{v^-}^-}{2} \Delta \Pi_{ab}
 *     + \frac{\lambda_{v^+}^- - \lambda_{v^-}^-}{2} n^j \Delta \Phi_{jab} \\
 *   D_{\Phi_{iab}}
 *     &= \lambda_{v^0}^- (\delta_i^j - n_i n^j) \Delta \Phi_{jab}
 *     + \frac{\lambda_{v^+}^- - \lambda_{v^-}^-}{2} n_i
 *       (\Delta \Pi_{ab} - \gamma_2 \Delta g_{ab})
 *     + \frac{\lambda_{v^+}^- + \lambda_{v^-}^-}{2} n_i n^j \Delta \Phi_{jab}.
 * \f}
 *
 * In the above expressions, the outgoing face normal $n_i$ is
 * normalized using the average metric from the two sides.  Quantities
 * such as $\gamma_2$ that should analytically be the same on both
 * sides of the interface are also averaged, as they can differ
 * because of truncation errors during projection.
 *
 * The characteristic speeds are given by
 *
 * \f{align}{
 *   \lambda_{v^g} &= - (1 + \gamma_1) (\beta^i + v^i_g) n_i, \\
 *   \lambda_{v^0} &= - (\beta^i + v^i_g) n_i, \\
 *   \lambda_{v^\pm} &= \pm \alpha - (\beta^i + v^i_g) n_i,
 * \f}
 *
 * where \f$v_g^i\f$ is the mesh velocity, with the incoming speeds
 *
 * \f{equation}{
 *   \lambda^-_{\hat{\alpha}} =
 *     \begin{cases}
 *       \lambda_{\hat{\alpha}} & \lambda_{\hat{\alpha}} < 0 \\
 *       0 & \text{otherwise.}
 *     \end{cases}
 * \f}
 */
template <size_t Dim>
class AveragedUpwindPenalty final : public evolution::BoundaryCorrection {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Computes the upwind penalty boundary correction term for the "
      "generalized harmonic system using the averaged fields from the two "
      "sides of the interface."};

  AveragedUpwindPenalty() = default;
  AveragedUpwindPenalty(const AveragedUpwindPenalty&) = default;
  AveragedUpwindPenalty& operator=(const AveragedUpwindPenalty&) = default;
  AveragedUpwindPenalty(AveragedUpwindPenalty&&) = default;
  AveragedUpwindPenalty& operator=(AveragedUpwindPenalty&&) = default;
  ~AveragedUpwindPenalty() override = default;

  /// \cond
  explicit AveragedUpwindPenalty(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(AveragedUpwindPenalty);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection> get_clone() const override;

  struct MeshVelocity : db::SimpleTag {
    using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
  };

  using dg_package_field_tags =
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, Dim>,
                 gh::Tags::Pi<DataVector, Dim>, gh::Tags::Phi<DataVector, Dim>,
                 gh::Tags::ConstraintGamma1, gh::Tags::ConstraintGamma2,
                 domain::Tags::UnnormalizedFaceNormal<Dim>, MeshVelocity>;
  // We don't need lapse and shift, but this list must be exactly this
  // or the boundary condition code doesn't compile.
  using dg_package_data_temporary_tags =
      tmpl::list<gh::Tags::ConstraintGamma1, gh::Tags::ConstraintGamma2,
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, Dim>>;
  using dg_package_data_primitive_tags = tmpl::list<>;
  using dg_package_data_volume_tags = tmpl::list<>;
  using dg_boundary_terms_volume_tags = tmpl::list<>;

  double dg_package_data(
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          packaged_spacetime_metric,
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*> packaged_pi,
      gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*> packaged_phi,
      gsl::not_null<Scalar<DataVector>*> packaged_constraint_gamma1,
      gsl::not_null<Scalar<DataVector>*> packaged_constraint_gamma2,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> packaged_normal,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          packaged_mesh_velocity,

      const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,

      const Scalar<DataVector>& constraint_gamma1,
      const Scalar<DataVector>& constraint_gamma2,
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity) const;

  void dg_boundary_terms(
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          boundary_correction_spacetime_metric,
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          boundary_correction_pi,
      gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
          boundary_correction_phi,

      const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric_int,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi_int,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi_int,
      const Scalar<DataVector>& constraint_gamma1_int,
      const Scalar<DataVector>& constraint_gamma2_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_int,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& mesh_velocity_int,

      const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric_ext,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi_ext,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi_ext,
      const Scalar<DataVector>& constraint_gamma1_ext,
      const Scalar<DataVector>& constraint_gamma2_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_ext,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& mesh_velocity_ext,

      dg::Formulation dg_formulation) const;
};

template <size_t Dim>
bool operator==(const AveragedUpwindPenalty<Dim>& lhs,
                const AveragedUpwindPenalty<Dim>& rhs);
template <size_t Dim>
bool operator!=(const AveragedUpwindPenalty<Dim>& lhs,
                const AveragedUpwindPenalty<Dim>& rhs);
}  // namespace gh::BoundaryCorrections
