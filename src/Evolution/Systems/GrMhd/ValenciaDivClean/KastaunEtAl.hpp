// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <string>

#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveRecoveryData.hpp"

/// \cond
namespace EquationsOfState {
template <bool, size_t>
class EquationOfState;
}  // namespace EquationsOfState
/// \endcond

namespace grmhd::ValenciaDivClean::PrimitiveRecoverySchemes {

/*!
 * \brief Compute the primitive variables from the conservative variables using
 * the scheme of \cite Kastaun2020uxr.
 *
 * In the notation of the Kastaun paper, `total_energy_density` is \f$D
 * (1+q)\f$, `momentum_density_squared` is \f$r^2 D^2\f$,
 * `momentum_density_dot_magnetic_field` is \f$t D^{\frac{3}{2}}\f$,
 * `magnetic_field_squared` is \f$s D\f$, and
 * `rest_mass_density_times_lorentz_factor` is \f$D\f$.
 * Furthermore, the returned `PrimitiveRecoveryData.rho_h_w_squared` is \f$x
 * D\f$.
 *
 * In terms of the conservative variables (in our notation):
 * \f{align*}
 * q = & \frac{{\tilde \tau}}{{\tilde D}} \\
 * r = & \frac{\gamma^{kl} {\tilde S}_k {\tilde S}_l}{{\tilde D}^2} \\
 * t^2 = & \frac{({\tilde B}^k {\tilde S}_k)^2}{{\tilde D}^3 \sqrt{\gamma}} \\
 * s = & \frac{\gamma_{kl} {\tilde B}^k {\tilde B}^l}{{\tilde D}\sqrt{\gamma}}
 * \f}
 *
 * where the conserved variables \f${\tilde D}\f$, \f${\tilde S}_i\f$,
 * \f${\tilde \tau}\f$, and \f${\tilde B}^i\f$ are a generalized mass-energy
 * density, momentum density, specific internal energy density, and magnetic
 * field, and \f$\gamma\f$ and \f$\gamma^{kl}\f$ are the determinant and inverse
 * of the spatial metric \f$\gamma_{kl}\f$.
 *
 * \note This scheme does not use the initial guess for the pressure.
 */
class KastaunEtAl {
 public:
  template <size_t ThermodynamicDim>
  static std::optional<PrimitiveRecoveryData> apply(
      double initial_guess_pressure, double total_energy_density,
      double momentum_density_squared,
      double momentum_density_dot_magnetic_field, double magnetic_field_squared,
      double rest_mass_density_times_lorentz_factor,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
          equation_of_state);

  static const std::string name() { return "KastaunEtAl"; }

 private:
  static constexpr size_t max_iterations_ = 100;
  static constexpr double absolute_tolerance_ =
      10.0 * std::numeric_limits<double>::epsilon();
  static constexpr double relative_tolerance_ =
      10.0 * std::numeric_limits<double>::epsilon();
};
}  // namespace grmhd::ValenciaDivClean::PrimitiveRecoverySchemes
