// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
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

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
/*!
 * \brief A Rusanov/local Lax-Friedrichs Riemann solver
 *
 * Let \f$U\f$ be the state vector of evolved variables, \f$F^i\f$ the
 * corresponding fluxes, and \f$n_i\f$ be the outward directed unit normal to
 * the interface. Denoting \f$F := n_i F^i\f$, the %Rusanov boundary correction
 * is
 *
 * \f{align*}
 * G_\text{Rusanov} = \frac{F_\text{int} - F_\text{ext}}{2} -
 * \frac{\text{max}\left(\{|\lambda_\text{int}|\},
 * \{|\lambda_\text{ext}|\}\right)}{2} \left(U_\text{ext} - U_\text{int}\right),
 * \f}
 *
 * where "int" and "ext" stand for interior and exterior, and
 * \f$\{|\lambda|\}\f$ is the set of characteristic/signal speeds. The minus
 * sign in front of the \f$F_{\text{ext}}\f$ is necessary because the outward
 * directed normal of the neighboring element has the opposite sign, i.e.
 * \f$n_i^{\text{ext}}=-n_i^{\text{int}}\f$. The characteristic/signal speeds
 * are given in the documentation for
 * grmhd::ValenciaDivClean::characteristic_speeds(). Since the fluid is
 * travelling slower than the speed of light, the speeds we are interested in
 * are
 *
 * \f{align*}{
 *   \lambda_{\pm}&=\pm\alpha-\beta^i n_i,
 * \f}
 *
 * which correspond to the divergence cleaning field.
 *
 * \note
 * - In the strong form the `dg_boundary_terms` function returns
 * \f$G - F_\text{int}\f$
 * - It may be possible to use the slower speeds for the magnetic field and
 * fluid part of the system in order to make the flux less dissipative for
 * those variables.
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
      "for the GRMHD system."};

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
      tmpl::list<Tags::TildeD, Tags::TildeTau, Tags::TildeS<Frame::Inertial>,
                 Tags::TildeB<Frame::Inertial>, Tags::TildePhi,
                 ::Tags::NormalDotFlux<Tags::TildeD>,
                 ::Tags::NormalDotFlux<Tags::TildeTau>,
                 ::Tags::NormalDotFlux<Tags::TildeS<Frame::Inertial>>,
                 ::Tags::NormalDotFlux<Tags::TildeB<Frame::Inertial>>,
                 ::Tags::NormalDotFlux<Tags::TildePhi>, AbsCharSpeed>;
  using dg_package_data_temporary_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<3, Frame::Inertial, DataVector>>;
  using dg_package_data_primitive_tags = tmpl::list<>;
  using dg_package_data_volume_tags = tmpl::list<>;

  static double dg_package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_tilde_d,
      gsl::not_null<Scalar<DataVector>*> packaged_tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> packaged_tilde_s,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> packaged_tilde_b,
      gsl::not_null<Scalar<DataVector>*> packaged_tilde_phi,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_d,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          packaged_normal_dot_flux_tilde_s,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          packaged_normal_dot_flux_tilde_b,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_phi,
      gsl::not_null<Scalar<DataVector>*> packaged_abs_char_speed,

      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      const Scalar<DataVector>& tilde_phi,

      const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_d,
      const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_tau,
      const tnsr::Ij<DataVector, 3, Frame::Inertial>& flux_tilde_s,
      const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_tilde_b,
      const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_phi,

      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& shift,

      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity);

  static void dg_boundary_terms(
      gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_d,
      gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          boundary_correction_tilde_s,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          boundary_correction_tilde_b,
      gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_phi,
      const Scalar<DataVector>& tilde_d_int,
      const Scalar<DataVector>& tilde_tau_int,
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_int,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_int,
      const Scalar<DataVector>& tilde_phi_int,
      const Scalar<DataVector>& normal_dot_flux_tilde_d_int,
      const Scalar<DataVector>& normal_dot_flux_tilde_tau_int,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_tilde_s_int,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_tilde_b_int,
      const Scalar<DataVector>& normal_dot_flux_tilde_phi_int,
      const Scalar<DataVector>& abs_char_speed_int,
      const Scalar<DataVector>& tilde_d_ext,
      const Scalar<DataVector>& tilde_tau_ext,
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_ext,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_ext,
      const Scalar<DataVector>& tilde_phi_ext,
      const Scalar<DataVector>& normal_dot_flux_tilde_d_ext,
      const Scalar<DataVector>& normal_dot_flux_tilde_tau_ext,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_tilde_s_ext,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_tilde_b_ext,
      const Scalar<DataVector>& normal_dot_flux_tilde_phi_ext,
      const Scalar<DataVector>& abs_char_speed_ext,
      dg::Formulation dg_formulation);
};
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
