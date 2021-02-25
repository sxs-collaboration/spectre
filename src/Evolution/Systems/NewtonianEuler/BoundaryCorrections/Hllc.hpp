// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
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

namespace NewtonianEuler::BoundaryCorrections {
/*!
 * \brief The HLLC (Harten-Lax-van Leer-Contact) Riemann solver for the
 * NewtonianEuler system
 *
 * Let \f$U\f$ be the evolved variable, \f$F^i\f$ the flux, \f$v^i\f$ the
 * spatial velocity, and \f$n_i\f$ be the outward directed unit normal to the
 * interface. Denoting \f$F := n_i F^i\f$ and \f$v:=n_iv^i\f$, the HLLC boundary
 * correction is \cite Toro1994
 *
 * \f{align*}
 * G_\text{HLLC} = \left\{\begin{array}{lcl}
 * F_\text{int} & \text{if} & 0 \leq \lambda_\text{min} \\
 * F_\text{int} + \lambda_\text{min}(U_\text{*int} - U_\text{int}) & \text{if} &
 * \lambda_\text{min} \leq 0 \leq \lambda_\text{*} \\
 * -F_\text{ext} + \lambda_\text{max}(U_\text{*ext} - U_\text{ext}) & \text{if}
 * & \lambda_\text{*} \leq 0 \leq \lambda_\text{max} \\
 * -F_\text{ext} & \text{if} &  \lambda_\text{max} \leq 0
 * \end{array}\right\}
 * \f}
 *
 * where "int" and "ext" stand for interior and exterior.
 *
 * Intermediate ('star') states are given by
 *
 * \f{align*}
 * U_\text{*int} = \left(\frac{\lambda_\text{min} - v_\text{int}}
 * {\lambda_\text{min} - \lambda_*}\right)
 * \left[\begin{array}{c}
 * \displaystyle \rho_\text{int} \\
 * \displaystyle \rho_\text{int}[v_\text{int}^x + (\lambda_* - v_\text{int})
 * n_x^\text{int}] \\
 * \displaystyle \rho_\text{int}[v_\text{int}^y + (\lambda_* - v_\text{int})
 * n_y^\text{int}] \\
 * \displaystyle \rho_\text{int}[v_\text{int}^z + (\lambda_* - v_\text{int})
 * n_z^\text{int}] \\
 * \displaystyle E_\text{int} + p_\text{int} \frac{\lambda_* - v_\text{int}}
 * {\lambda_\text{min} - v_\text{int}} + \rho_\text{int}\lambda_*(\lambda_* -
 * v_\text{int}) \end{array}\right]
 * \f}
 *
 * and
 *
 * \f{align*}
 * U_\text{*ext} = \left(\frac{\lambda_\text{max} + v_\text{ext}}
 * {\lambda_\text{max} - \lambda_*}\right)
 * \left[\begin{array}{c}
 * \displaystyle \rho_\text{ext} \\
 * \displaystyle \rho_\text{ext}[-v_\text{ext}^x - (\lambda_* + v_\text{ext})
 * n_x^\text{ext}] \\
 * \displaystyle \rho_\text{ext}[-v_\text{ext}^y - (\lambda_* + v_\text{ext})
 * n_y^\text{ext}] \\
 * \displaystyle \rho_\text{ext}[-v_\text{ext}^z - (\lambda_* + v_\text{ext})
 * n_z^\text{ext}] \\
 * \displaystyle E_\text{ext} + p_\text{ext} \frac{\lambda_* + v_\text{ext}}
 * {\lambda_\text{max} + v_\text{ext}} + \rho_\text{ext}\lambda_*(\lambda_* +
 * v_\text{ext}) \end{array}\right].
 * \f}
 *
 * The contact wave speed \f$\lambda_*\f$ is \cite Toro2009
 *
 * \f{align*}
 * \lambda_* = \frac
 * { p_\text{ext} - p_\text{int} +
 * \rho_\text{int}v_\text{int}(\lambda_\text{min}-v_\text{int})
 * + \rho_\text{ext}v_\text{ext}(\lambda_\text{max}+v_\text{ext})}
 * { \rho_\text{int}(\lambda_\text{min}-v_\text{int}) -
 * \rho_\text{ext}(\lambda_\text{max} + v_\text{ext})}.
 * \f}
 *
 * \f$\lambda_\text{min}\f$ and \f$\lambda_\text{max}\f$ are estimated by
 * \cite Davis1988
 *
 * \f{align*}
 * \lambda_\text{min} &=
 * \text{min}\left(\lambda^{-}_\text{int},-\lambda^{+}_\text{ext}\right) \\
 * \lambda_\text{max} &=
 * \text{max}\left(\lambda^{+}_\text{int},-\lambda^{-}_\text{ext}\right)
 * \f}
 *
 * where \f$\lambda^{+}\f$ (\f$\lambda^{-}\f$) is the largest characteristic
 * speed in the outgoing (ingoing) direction for each domain.
 *
 * Note the minus signs in front of \f$\lambda^{\pm}_\text{ext}\f$, which is
 * because an outgoing speed w.r.t. the neighboring element is an ingoing speed
 * w.r.t. the local element, and vice versa. Similarly, the \f$F_{\text{ext}}\f$
 * term in \f$G_\text{HLLC}\f$ and the \f$v_\text{ext}\f$ term in
 * \f$U_\text{*ext}\f$ have a positive sign because the outward directed normal
 * of the neighboring element has the opposite sign, i.e.
 * \f$n_i^{\text{ext}}=-n_i^{\text{int}}\f$.
 *
 * For the NewtonianEuler system, \f$\lambda^\pm\f$ are given as
 *
 * \f{align*}
 * \lambda^\pm = v \pm c_s
 * \f}
 *
 * where \f$c_s\f$ is the sound speed.
 *
 * \note
 * - In the strong form the `dg_boundary_terms` function returns \f$G -
 *   F_\text{int}\f$
 * - In the implementation, we use
 *
 *   \f{align*}
 *   G_\text{HLLC} = \left\{\begin{array}{lcl}
 *   F_\text{int} + \lambda_\text{min}(U_\text{*int} - U_\text{int}) & \text{if}
 *   & 0 \leq \lambda_\text{*} \\
 *   -F_\text{ext} + \lambda_\text{max}(U_\text{*ext} - U_\text{ext}) &
 *   \text{if} & \lambda_\text{*} \leq 0 \\
 *   \end{array}\right\},
 *   \f}
 *
 *   with
 *
 *   \f{align*}
 *   \lambda_\text{min} &=
 *   \text{min}\left(\lambda^{-}_\text{int},-\lambda^{+}_\text{ext}, 0\right) \\
 *   \lambda_\text{max} &=
 *   \text{max}\left(\lambda^{+}_\text{int},-\lambda^{-}_\text{ext}, 0\right).
 *   \f}
 *
 *   Provided that \f$\lambda_*\f$ falls in the correct range i.e
 *
 *   \f{align*}
 *   \text{min}\left(\lambda^{-}_\text{int},-\lambda^{+}_\text{ext}\right)
 *   < \lambda_* <
 *   \text{max}\left(\lambda^{+}_\text{int},-\lambda^{-}_\text{ext}\right),
 *   \f}
 *
 *   this prescription recovers the original HLLC boundary correction for all
 *   four cases. For either \f$\lambda_\text{min} = 0\f$ or
 *   \f$\lambda_\text{max} = 0\f$ (i.e. all characteristics move in the same
 *   direction), boundary correction reduces to pure upwinding.
 * - Some references use \f$S\f$ instead of \f$\lambda\f$ for the
 *   signal/characteristic speeds.
 */
template <size_t Dim>
class Hllc final : public BoundaryCorrection<Dim> {
 private:
  struct InterfaceUnitNormal : db::SimpleTag {
    using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
  };
  struct NormalDotVelocity : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

 public:
  struct LargestOutgoingCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct LargestIngoingCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Computes the HLLC boundary correction term for the "
      "Newtonian Euler/hydrodynamics system."};

  Hllc() = default;
  Hllc(const Hllc&) = default;
  Hllc& operator=(const Hllc&) = default;
  Hllc(Hllc&&) = default;
  Hllc& operator=(Hllc&&) = default;
  ~Hllc() override = default;

  /// \cond
  explicit Hllc(CkMigrateMessage* msg) noexcept;
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Hllc);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection<Dim>> get_clone() const noexcept override;

  using dg_package_field_tags = tmpl::list<
      Tags::MassDensityCons, Tags::MomentumDensity<Dim>, Tags::EnergyDensity,
      Tags::Pressure<DataVector>, ::Tags::NormalDotFlux<Tags::MassDensityCons>,
      ::Tags::NormalDotFlux<Tags::MomentumDensity<Dim>>,
      ::Tags::NormalDotFlux<Tags::EnergyDensity>, InterfaceUnitNormal,
      NormalDotVelocity, LargestOutgoingCharSpeed, LargestIngoingCharSpeed>;
  using dg_package_data_temporary_tags = tmpl::list<>;
  using dg_package_data_primitive_tags =
      tmpl::list<NewtonianEuler::Tags::Velocity<DataVector, Dim>,
                 NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>>;
  using dg_package_data_volume_tags =
      tmpl::list<hydro::Tags::EquationOfStateBase>;

  template <size_t ThermodynamicDim>
  double dg_package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_mass_density,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          packaged_momentum_density,
      gsl::not_null<Scalar<DataVector>*> packaged_energy_density,
      gsl::not_null<Scalar<DataVector>*> packaged_pressure,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_mass_density,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          packaged_normal_dot_flux_momentum_density,
      gsl::not_null<Scalar<DataVector>*>
          packaged_normal_dot_flux_energy_density,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_interface_unit_normal,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_velocity,
      gsl::not_null<Scalar<DataVector>*> packaged_largest_outgoing_char_speed,
      gsl::not_null<Scalar<DataVector>*> packaged_largest_ingoing_char_speed,

      const Scalar<DataVector>& mass_density,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& momentum_density,
      const Scalar<DataVector>& energy_density,

      const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_mass_density,
      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_momentum_density,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_energy_density,

      const tnsr::I<DataVector, Dim, Frame::Inertial>& velocity,
      const Scalar<DataVector>& specific_internal_energy,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
          equation_of_state) const noexcept;

  void dg_boundary_terms(
      gsl::not_null<Scalar<DataVector>*> boundary_correction_mass_density,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          boundary_correction_momentum_density,
      gsl::not_null<Scalar<DataVector>*> boundary_correction_energy_density,
      const Scalar<DataVector>& mass_density_int,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& momentum_density_int,
      const Scalar<DataVector>& energy_density_int,
      const Scalar<DataVector>& pressure_int,
      const Scalar<DataVector>& normal_dot_flux_mass_density_int,
      const tnsr::I<DataVector, Dim, Frame::Inertial>&
          normal_dot_flux_momentum_density_int,
      const Scalar<DataVector>& normal_dot_flux_energy_density_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          interface_unit_normal_int,
      const Scalar<DataVector>& normal_dot_velocity_int,
      const Scalar<DataVector>& largest_outgoing_char_speed_int,
      const Scalar<DataVector>& largest_ingoing_char_speed_int,
      const Scalar<DataVector>& mass_density_ext,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& momentum_density_ext,
      const Scalar<DataVector>& energy_density_ext,
      const Scalar<DataVector>& pressure_ext,
      const Scalar<DataVector>& normal_dot_flux_mass_density_ext,
      const tnsr::I<DataVector, Dim, Frame::Inertial>&
          normal_dot_flux_momentum_density_ext,
      const Scalar<DataVector>& normal_dot_flux_energy_density_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          interface_unit_normal_ext,
      const Scalar<DataVector>& normal_dot_velocity_ext,
      const Scalar<DataVector>& largest_outgoing_char_speed_ext,
      const Scalar<DataVector>& largest_ingoing_char_speed_ext,
      dg::Formulation dg_formulation) const noexcept;
};
}  // namespace NewtonianEuler::BoundaryCorrections
