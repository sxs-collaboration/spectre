// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ForceFree/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
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

namespace ForceFree::BoundaryCorrections {

/*!
 * \brief A Rusanov/local Lax-Friedrichs Riemann solver
 *
 * Let \f$U\f$ be the evolved variables, \f$F^i\f$ the corresponding fluxes, and
 * \f$n_i\f$ be the outward directed unit normal to the interface. Denoting \f$F
 * := n_i F^i\f$, the %Rusanov boundary correction is
 *
 * \f{align*}
 * G_\text{Rusanov} = \frac{F_\text{int} - F_\text{ext}}{2} -
 * \frac{\text{max}\left(|\lambda_\text{int}|,
 * |\lambda_\text{ext}|\right)}{2} \left(U_\text{ext} - U_\text{int}\right),
 * \f}
 *
 * where "int" and "ext" stand for interior and exterior, and
 * \f$\lambda\f$ is the characteristic/signal speed. The minus sign in
 * front of the \f$F_{\text{ext}}\f$ is necessary because the outward directed
 * normal of the neighboring element has the opposite sign, i.e.
 * \f$n_i^{\text{ext}}=-n_i^{\text{int}}\f$.
 *
 * For the GRFFE system the largest characteristic speeds \f$\lambda\f$ of our
 * interest are given as
 *
 * \f{align*}{
 *   \lambda_{\pm} = -\beta^i n_i \pm \alpha.
 * \f}
 *
 * which correspond to fast mode waves.
 *
 * \note In the strong form the `dg_boundary_terms` function returns
 * \f$G - F_\text{int}\f$
 *
 */
class Rusanov final : public BoundaryCorrection {
 private:
  struct AbsCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Computes the Rusanov or local Lax-Friedrichs boundary correction term "
      "for the GRFFE system."};

  Rusanov() = default;
  Rusanov(const Rusanov&) = default;
  Rusanov& operator=(const Rusanov&) = default;
  Rusanov(Rusanov&&) = default;
  Rusanov& operator=(Rusanov&&) = default;
  ~Rusanov() override = default;

  /// \cond
  explicit Rusanov(CkMigrateMessage* /*unused*/);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Rusanov);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection> get_clone() const override;

  using dg_package_field_tags =
      tmpl::list<Tags::TildeE, Tags::TildeB, Tags::TildePsi, Tags::TildePhi,
                 Tags::TildeQ, ::Tags::NormalDotFlux<Tags::TildeE>,
                 ::Tags::NormalDotFlux<Tags::TildeB>,
                 ::Tags::NormalDotFlux<Tags::TildePsi>,
                 ::Tags::NormalDotFlux<Tags::TildePhi>,
                 ::Tags::NormalDotFlux<Tags::TildeQ>, AbsCharSpeed>;
  using dg_package_data_temporary_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>>;
  using dg_package_data_primitive_tags = tmpl::list<>;
  using dg_package_data_volume_tags = tmpl::list<>;

  static double dg_package_data(
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> packaged_tilde_e,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> packaged_tilde_b,
      gsl::not_null<Scalar<DataVector>*> packaged_tilde_psi,
      gsl::not_null<Scalar<DataVector>*> packaged_tilde_phi,
      gsl::not_null<Scalar<DataVector>*> packaged_tilde_q,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          packaged_normal_dot_flux_tilde_e,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          packaged_normal_dot_flux_tilde_b,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_psi,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_phi,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_q,
      gsl::not_null<Scalar<DataVector>*> packaged_abs_char_speed,

      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      const Scalar<DataVector>& tilde_psi, const Scalar<DataVector>& tilde_phi,
      const Scalar<DataVector>& tilde_q,

      const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_tilde_e,
      const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_tilde_b,
      const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_psi,
      const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_phi,
      const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_q,

      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& shift,

      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity);

  static void dg_boundary_terms(
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          boundary_correction_tilde_e,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          boundary_correction_tilde_b,
      gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_psi,
      gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_phi,
      gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_q,

      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e_int,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_int,
      const Scalar<DataVector>& tilde_psi_int,
      const Scalar<DataVector>& tilde_phi_int,
      const Scalar<DataVector>& tilde_q_int,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_tilde_e_int,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_tilde_b_int,
      const Scalar<DataVector>& normal_dot_flux_tilde_psi_int,
      const Scalar<DataVector>& normal_dot_flux_tilde_phi_int,
      const Scalar<DataVector>& normal_dot_flux_tilde_q_int,
      const Scalar<DataVector>& abs_char_speed_int,

      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e_ext,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_ext,
      const Scalar<DataVector>& tilde_psi_ext,
      const Scalar<DataVector>& tilde_phi_ext,
      const Scalar<DataVector>& tilde_q_ext,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_tilde_e_ext,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_tilde_b_ext,
      const Scalar<DataVector>& normal_dot_flux_tilde_psi_ext,
      const Scalar<DataVector>& normal_dot_flux_tilde_phi_ext,
      const Scalar<DataVector>& normal_dot_flux_tilde_q_ext,
      const Scalar<DataVector>& abs_char_speed_ext,
      dg::Formulation dg_formulation);
};

bool operator==(const Rusanov& lhs, const Rusanov& rhs);
bool operator!=(const Rusanov& lhs, const Rusanov& rhs);

}  // namespace ForceFree::BoundaryCorrections
