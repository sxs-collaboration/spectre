// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace grmhd {
namespace ValenciaDivClean {

/// \brief Schemes for recovering primitive variables from conservative
/// variables.
namespace PrimitiveRecoverySchemes {

/*!
 * \brief Data determined by PrimitiveRecoverySchemes at a single grid point.
 *
 * `rho_h_w_squared` is \f$\rho h W^2\f$ where \f$\rho\f$ is the rest mass
 * density, \f$h\f$ is the specific enthalpy, and \f$W\f$ is the Lorentz factor.
 */
struct PrimitiveRecoveryData {
  double rest_mass_density;
  double lorentz_factor;
  double pressure;
  double rho_h_w_squared;
};
}  // namespace PrimitiveRecoverySchemes
}  // namespace ValenciaDivClean
}  // namespace grmhd
