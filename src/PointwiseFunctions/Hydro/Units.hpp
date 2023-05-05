// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>

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
 */
namespace cgs {

/// The speed of light in cm/s (This is exact)
constexpr double speed_of_light = 29979245800.0;
/// Newton's gravitational constant as given by
/// https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.93.025010.
/// G is likely to vary between sources, even beyond the reported uncertainty
constexpr double G_Newton = 6.67430e-8;
/// Note G*M_Sun = 1.32712440042x10^26 +/- 1e16 [1/cm]
/// G*M_Sun factor from https://doi.org/10.1063/1.4921980
static constexpr double G_Newton_times_m_sun = 1.32712440042e26;
/// k_B factor in erg/K from
/// https://journals.aps.org/rmp/pdf/10.1103/RevModPhys.93.025010.
/// (This is exact)
static constexpr double k_Boltzmann = 1.380649e-16;

/// `mass_cgs = mass_geometric * mass_unit`
/// Heuristically the mass of the sun in grams
constexpr double mass_unit = G_Newton_times_m_sun / G_Newton;
/// `length_cgs = length_geometric * length_unit`
/// Heuristically, half the schwarzschild radius of the sun
/// in cm
constexpr double length_unit = G_Newton_times_m_sun / square(speed_of_light);
/// `time_cgs = time_geometric * time_unit`
/// Heuristically, half the light crossing time of a
/// solar schwarzschild radius in seconds
constexpr double time_unit = length_unit / speed_of_light;
/// `rho_cgs = rho_geometric * rho_unit`
/// Heuristically, the density in g/cm^3 of matter ~2200 times the density of
/// atomic nuclei.  Note this is much larger than any density realized in the
/// universe.
constexpr double rest_mass_density_unit =
    mass_unit / (length_unit * length_unit * length_unit);
/// `pressure_cgs = pressure_geometric * pressure_unit`
/// Heuristically, the quantity above but times the speed of light squared in
/// cgs
constexpr double pressure_unit =
    mass_unit / (length_unit * time_unit * time_unit);

// Extra constants, which are known to greatest precision in SI / (equvialently
// CGS)
/// The accepted value for the atomic mass unit in g, uncertainty +-30e-10
constexpr double atomic_mass_unit = 1.66053906660e-24;
/// The neutron mass, given in grams, uncertainty at 5.7 x 10^-10 level
constexpr double neutron_mass = 1.67492749804e-24;
/// The electron-volt (eV) given in ergs, which is known exactly in SI/cgs
constexpr double electron_volt = 1.602176634e-12;

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

/*!
 * \brief The defining quantities of the nuclear fm-MeV/c^2-fm/c unit system
 *
 * Constants are defined in terms of cgs units where possible, which are, in
 * many cases known exactly due to their relation to SI.
 */
namespace nuclear {
/// `mass_nuclear = mass_unit * mass_geometric`
/// Heuristically a solar mass in MeV/c^2
constexpr double mass_unit =
     cgs::mass_unit /
    (1.0e6 * cgs::electron_volt / square(cgs::speed_of_light));
/// `length_nuclear = length_unit * length_geometric`
/// Heuristically GM_sun/c^2 in fm
constexpr double length_unit =  cgs::length_unit / (1.0e-13);
/// `time_nuclear = time_unit * time_geometric`
/// Heuristically GM_sun/c^3 in fm/c
constexpr double time_unit =
     cgs::time_unit / (1.0e-13 / cgs::speed_of_light);
/// `pressure_nuclear = pressure_geometric * pressure_geometric`
/// Heuristically the rest-mass energy density of
/// ~2200 the density of atomic nucli expressed in MeV/fm^3
constexpr double pressure_unit = mass_unit / (length_unit * square(time_unit));
/// `rest_mass_density_nuclear = rest_mass_density_geometric *
/// rest_mass_density_geometric` Heuristically ~2200 the density of atomic nucli
/// expressed in MeV/c^2/fm^3
constexpr double rest_mass_density_unit = mass_unit / cube(length_unit);
constexpr double neutron_mass =
    cgs::neutron_mass /
    (1.0e6 * cgs::electron_volt / square(cgs::speed_of_light));
constexpr double atomic_mass_unit =
    cgs::atomic_mass_unit /
    (1.0e6 * cgs::electron_volt / square(cgs::speed_of_light));

/// The saturation number density of baryons in nuclear matter, in
/// units of 1/fm^3.  This is a standard value, consistent with  e.g.
/// https://journals.aps.org/prc/abstract/10.1103/PhysRevC.102.044321
constexpr double saturation_number_density = 0.16;

}  // namespace nuclear
/*!
 * \brief Quantities given in terms of geometric units G = c = M_odot =
 * 1
 *
 * All quantities are given in terms of unit systems which are related exactly
 * to SI units.  The limiting factor in conversions is the poorly known value
 * of G; which leads to large uncertainties in the relative value of SI-derived
 * masses to the solar mass.  For this reason, it is best to avoid using
 * geometric units in calculations except where conversion is explicitly needed.
 */
namespace geometric {
/// Neutron mass in G = c = M_sun = 1.0 units
constexpr double neutron_mass = cgs::neutron_mass / cgs::mass_unit;
/// The rest mass density of matter at the saturation point
constexpr double default_saturation_rest_mass_density =
    1 / nuclear::rest_mass_density_unit *
    (nuclear::saturation_number_density * nuclear::atomic_mass_unit);
/// The default baryon mass in geometric units.
/// Accepted value of atomic mass unit as expressed in
/// solar masses
/// Limited by precision of knowledge of solar mass.
constexpr double default_baryon_mass = cgs::atomic_mass_unit / cgs::mass_unit;
}  // namespace geometric

}  // namespace units
}  // namespace hydro
