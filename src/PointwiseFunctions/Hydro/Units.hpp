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

/*!
 * \brief CGS unit of Gauss.
 *
 * Equals `mass_unit^(1/2) * length_unit^(-1/2) * time_unit^(-1)`.
 *
 * Conversion rule for electromagnetic fields is
 * ```
 * magnetic_field_cgs = magnetic_field_geometric * gauss_unit
 * ```
 * or
 * ```
 * electric_field_cgs = electric_field_geometric * gauss_unit
 * ```
 *
 * \warning Before changing units using this value, make sure the unit systems
 * you are converting values between are using the same unit convention for
 * electromagnetic variables as well. Gaussian and Heaviside-Lorentz unit
 * convention (in electromagnetism) have a factor of \f$\sqrt{4\pi}\f$
 * difference in variables. See
 * <a href="https://en.wikipedia.org/wiki/Heaviside%E2%80%93Lorentz_units">this
 * Wikipedia page</a> for how to convert between different unit systems in
 * electromagnetism.
 *
 * Note that ForceFree and grmhd::ValenciaDivClean evolution systems are
 * adopting the geometrized Heaviside-Lorentz unit for magnetic fields.
 *
 * e.g. Suppose magnetic field has value \f$10^{-5}\f$ with the code unit in the
 * ValenciaDivClean evolution system. This corresponds to
 * ```
 * 10^(-5) * gauss_unit = 2.3558985e+14 Gauss
 * ```
 * in the CGS Heaviside-Lorentz unit.
 *
 * If one wants to convert it to the usual CGS Gaussian unit, extra factor of
 * \f$\sqrt{4\pi}\f$ needs to be multiplied:
 * ```
 * 10^(-5) * gauss_unit * sqrt(4pi) = 8.35144274e+14 Gauss
 * ```
 */
constexpr double gauss_unit = 2.3558985e+19;
}  // namespace cgs
}  // namespace units
}  // namespace hydro
