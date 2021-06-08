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
 * \brief An HLL Riemann solver
 *
 * Let \f$U\f$ be the evolved variable, \f$F^i\f$ the flux, and \f$n_i\f$ be the
 * outward directed unit normal to the interface. Denoting \f$F := n_i F^i\f$,
 * the HLL boundary correction is \cite Harten1983
 *
 * \f{align*}
 * G_\text{HLL} = \frac{\lambda_\text{max} F_\text{int} +
 * \lambda_\text{min} F_\text{ext}}{\lambda_\text{max} - \lambda_\text{min}}
 * - \frac{\lambda_\text{min}\lambda_\text{max}}{\lambda_\text{max} -
 *   \lambda_\text{min}} \left(U_\text{int} - U_\text{ext}\right)
 * \f}
 *
 * where "int" and "ext" stand for interior and exterior.
 * \f$\lambda_\text{min}\f$ and \f$\lambda_\text{max}\f$ are defined as
 *
 * \f{align*}
 * \lambda_\text{min} &=
 * \text{min}\left(\lambda^{-}_\text{int},-\lambda^{+}_\text{ext}, 0\right) \\
 * \lambda_\text{max} &=
 * \text{max}\left(\lambda^{+}_\text{int},-\lambda^{-}_\text{ext}, 0\right)
 * \f}
 *
 * where \f$\lambda^{+}\f$ (\f$\lambda^{-}\f$) is the largest characteristic
 * speed in the outgoing (ingoing) direction. Note the minus signs in front of
 * \f$\lambda^{\pm}_\text{ext}\f$, which is because an outgoing speed w.r.t. the
 * neighboring element is an ingoing speed w.r.t. the local element, and vice
 * versa. Similarly, the \f$F_{\text{ext}}\f$ term in \f$G_\text{HLL}\f$ has a
 * positive sign because the outward directed normal of the neighboring element
 * has the opposite sign, i.e. \f$n_i^{\text{ext}}=-n_i^{\text{int}}\f$.
 *
 * The characteristic/signal speeds are given in the documentation for
 * `grmhd::ValenciaDivClean::characteristic_speeds()`. Since the fluid is
 * travelling slower than the speed of light, the speeds we are interested in
 * are
 *
 * \f{align*}{
 *   \lambda^{\pm}&=\pm\alpha-\beta^i n_i,
 * \f}
 *
 * which correspond to the divergence cleaning field.
 *
 * \note
 * - In the strong form the `dg_boundary_terms` function returns
 *   \f$G - F_\text{int}\f$
 * - For either \f$\lambda_\text{min} = 0\f$ or \f$\lambda_\text{max} = 0\f$
 *   (i.e. all characteristics move in the same direction) the HLL boundary
 *   correction reduces to pure upwinding.
 * - Some references use \f$S\f$ instead of \f$\lambda\f$ for the
 *   signal/characteristic speeds
 * - It may be possible to use the slower speeds for the magnetic field and
 *   fluid part of the system in order to make the flux less dissipative for
 *   those variables.
 */
class Hll final : public BoundaryCorrection {
 public:
  struct LargestOutgoingCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct LargestIngoingCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Computes the HLL boundary correction term for the GRMHD system."};

  Hll() = default;
  Hll(const Hll&) = default;
  Hll& operator=(const Hll&) = default;
  Hll(Hll&&) = default;
  Hll& operator=(Hll&&) = default;
  ~Hll() override = default;

  /// \cond
  explicit Hll(CkMigrateMessage* /*unused*/) noexcept;
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Hll);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection> get_clone() const noexcept override;

  using dg_package_field_tags =
      tmpl::list<Tags::TildeD, Tags::TildeTau, Tags::TildeS<Frame::Inertial>,
                 Tags::TildeB<Frame::Inertial>, Tags::TildePhi,
                 ::Tags::NormalDotFlux<Tags::TildeD>,
                 ::Tags::NormalDotFlux<Tags::TildeTau>,
                 ::Tags::NormalDotFlux<Tags::TildeS<Frame::Inertial>>,
                 ::Tags::NormalDotFlux<Tags::TildeB<Frame::Inertial>>,
                 ::Tags::NormalDotFlux<Tags::TildePhi>,
                 LargestOutgoingCharSpeed, LargestIngoingCharSpeed>;
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
      gsl::not_null<Scalar<DataVector>*> packaged_largest_outgoing_char_speed,
      gsl::not_null<Scalar<DataVector>*> packaged_largest_ingoing_char_speed,

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
      const std::optional<Scalar<DataVector>>&
          normal_dot_mesh_velocity) noexcept;

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
      const Scalar<DataVector>& largest_outgoing_char_speed_int,
      const Scalar<DataVector>& largest_ingoing_char_speed_int,
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
      const Scalar<DataVector>& largest_outgoing_char_speed_ext,
      const Scalar<DataVector>& largest_ingoing_char_speed_ext,
      dg::Formulation dg_formulation) noexcept;
};
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
