// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/optional.hpp>
#include <cstddef>
#include <limits>
#include <string>

#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState

namespace grmhd {
namespace ValenciaDivClean {
namespace PrimitiveRecoverySchemes {

/// \cond
struct PrimitiveRecoveryData;
/// \endcond

/*!
 * \brief Compute the primitive variables from the conservative variables using
 * the scheme of [Palenzuela et al, Phys. Rev. D 92, 044045
 * (2015)](https://doi.org/10.1103/PhysRevD.92.044045).
 *
 * In the notation of the Palenzuela paper, `total_energy_density` is \f$D
 * (1+q)\f$, `momentum_density_squared` is \f$r^2 D^2\f$,
 * `momentum_density_dot_magnetic_field` is \f$t D^{\frac{3}{2}}\f$,
 * `magnetic_field_squared` is \f$s D\f$, and
 * `rest_mass_density_times_lorentz_factor` is \f$D\f$.
 * Furthermore, the returned `PrimitiveRecoveryData.rho_h_w_squared` is \f$x
 * D\f$.  Note also that \f$h\f$ in the Palenzuela paper is the specific
 * enthalpy times the rest mass density.
 *
 * In terms of the conservative variables (in our notation):
 * \f{align*}
 * q = & \frac{{\tilde \tau}}{{\tilde D}} \\
 * r = & \frac{\gamma^{mn} {\tilde S}_m {\tilde S}_n}{{\tilde D}^2} \\
 * t^2 = & \frac{({\tilde B}^m {\tilde S}_m)^2}{{\tilde D}^3 \sqrt{\gamma}} \\
 * s = & \frac{\gamma_{mn} {\tilde B}^m {\tilde B}^n}{{\tilde D}\sqrt{\gamma}}
 * \f}
 *
 * where the conserved variables \f${\tilde D}\f$, \f${\tilde S}_i\f$,
 * \f${\tilde \tau}\f$, and \f${\tilde B}^i\f$ are a generalized mass-energy
 * density, momentum density, specific internal energy density, and magnetic
 * field, and \f$\gamma\f$ and \f$\gamma^{mn}\f$ are the determinant and inverse
 * of the spatial metric \f$\gamma_{mn}\f$.
 */
class PalenzuelaEtAl {
 public:
  template <size_t ThermodynamicDim>
  static boost::optional<PrimitiveRecoveryData> apply(
      double /*initial_guess_pressure*/, double total_energy_density,
      double momentum_density_squared,
      double momentum_density_dot_magnetic_field, double magnetic_field_squared,
      double rest_mass_density_times_lorentz_factor,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
          equation_of_state) noexcept;

  static const std::string name() noexcept { return "PalenzuelaEtAl"; }

 private:
  static constexpr size_t max_iterations_ = 100;
  static constexpr double absolute_tolerance_ =
      10.0 * std::numeric_limits<double>::epsilon();
  static constexpr double relative_tolerance_ =
      10.0 * std::numeric_limits<double>::epsilon();
};
}  // namespace PrimitiveRecoverySchemes
}  // namespace ValenciaDivClean
}  // namespace grmhd
