// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <string>


/// \cond
namespace EquationsOfState {
template <bool, size_t>
class EquationOfState;
}  // namespace EquationsOfState
namespace grmhd::ValenciaDivClean {
class PrimitiveFromConservativeOptions;
}  // namespace grmhd::ValenciaDivClean
/// \endcond

namespace grmhd::ValenciaDivClean::PrimitiveRecoverySchemes {

struct PrimitiveRecoveryData;

/*!
 * \brief Compute the primitive variables from the conservative variables using
 * the scheme of \cite Galeazzi2013mia.
 *
 * In the notation of the Kastaun paper, `tau` is \f$D q\f$,
 * `momentum_density_squared` is \f$r^2 D^2\f$,
 * `rest_mass_density_times_lorentz_factor` is \f$D\f$.
 * Furthermore, the algorithm iterates over \f$z\f$, which is the Lorentz factor
 * times the absolute magnetitude of the velocity.
 *
 * In terms of the conservative variables (in our notation):
 * \f{align*}
 * q = & \frac{{\tilde \tau}}{{\tilde D}} \\
 * r^2 = & \frac{\gamma^{kl} {\tilde S}_k {\tilde S}_l}{{\tilde D}^2} \\
 * \f}
 *
 * where the conserved variables \f${\tilde D}\f$, \f${\tilde S}_i\f$ and
 * \f${\tilde \tau}\f$ are a generalized mass-energy
 * density, momentum density and specific internal energy density,
 * and \f$\gamma\f$ and \f$\gamma^{kl}\f$ are the determinant and inverse
 * of the spatial metric \f$\gamma_{kl}\f$.
 *
 * \note This scheme does not use the initial guess for the pressure.
 */
class KastaunEtAlHydro {
 public:
  template <bool EnforcePhysicality, typename EosType>
  static std::optional<PrimitiveRecoveryData> apply(
      double initial_guess_pressure, double tau,
      double momentum_density_squared,
      double momentum_density_dot_magnetic_field, double magnetic_field_squared,
      double rest_mass_density_times_lorentz_factor, double electron_fraction,
      const EosType& equation_of_state,
      const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions&
          primitive_from_conservative_options);

  static const std::string name() { return "KastaunEtAlHydro"; }

 private:
  static constexpr size_t max_iterations_ = 100;
  static constexpr double absolute_tolerance_ =
      10.0 * std::numeric_limits<double>::epsilon();
  static constexpr double relative_tolerance_ =
      10.0 * std::numeric_limits<double>::epsilon();
};
}  // namespace grmhd::ValenciaDivClean::PrimitiveRecoverySchemes
