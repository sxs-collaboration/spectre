// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace hydro {
/*!
 * \brief Functions, constants, and classes for converting between different
 * units.
 *
 * In SpECTRE we prefer to use geometric units where \f$G=c=M_{\odot}=1\f$, as
 * is standard in numerical relativity. However, in order to interface with
 * other codes, we need to sometimes convert units. This namespace is designed
 * to hold the various conversion factors and functions. A unit library/system
 * would be nice, but is a non-trivial amount of work.
 */
namespace units {
/*!
 * \brief Entities for converting between geometric units where
 * \f$G=c=M_{\odot}=1\f$ and CGS units.
 *
 * Note: the baryon mass is not implemented here since it depends on what the
 * specific place assumes. E.g. different EoS can assume different baryon
 * masses. See https://github.com/sxs-collaboration/spectre/issues/4694
 */
namespace cgs {
/// `mass_cgs = mass_geometric * mass_unit`
constexpr double mass_unit = 1.0 / 5.02765209e-34;
/// `length_cgs = length_geometric * length_unit`
constexpr double length_unit = 1.0 / 6.77140812e-6;
/// `time_cgs = time_geometric * time_unit`
constexpr double time_unit = 1.0 / 2.03001708e5;
/// `rho_cgs = rho_geometric * rho_unit`
constexpr double rest_mass_density_unit =
    mass_unit / (length_unit * length_unit * length_unit);
/// `pressure_cgs = pressure_geometric * pressure_unit`
constexpr double pressure_unit =
    mass_unit / (length_unit * time_unit * time_unit);
}  // namespace cgs
}  // namespace units
}  // namespace hydro
