// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/MakeArray.hpp"            // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma:  no_include <pup.h>

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
/// \endcond

namespace grmhd {
namespace Solutions {

/*!
 * \brief Circularly polarized Alfv&eacute;n wave solution in Minkowski
 * spacetime travelling along a background magnetic field.
 *
 * An analytic solution to the 3-D GRMHD system. The user specifies the
 * wavenumber \f$k\f$ of the Alfv&eacute;n wave, the constant pressure
 * throughout the fluid \f$P\f$, the constant rest mass density throughout the
 * fluid \f$\rho_0\f$, the adiabatic index for the ideal fluid equation of
 * state \f$\gamma\f$, the magnetic field parallel to the wavevector
 * \f$\vec{B}_0\f$, and the transverse magnetic field vector \f$\vec{B}_1\f$ at
 * \f$x=y=z=t=0\f$.
 *
 * We define the auxiliary velocities:
 * \f[v^2_{B0} = \frac{B_0^2}{\rho_0 h + B_0^2 + B_1^2}\f]
 * \f[v^2_{B1} = \frac{B_1^2}{\rho_0 h + B_0^2 + B_1^2}\f]
 *
 * The Alfv&eacute;n wave phase speed that solves the GRMHD equations, even for
 * finite amplitudes \cite DelZanna2007pk, is given by:
 *
 * \f[v_A^2 = \frac{2v^2_{B0}}{1 + \sqrt{1 - 4 v^2_{B0}v^2_{B1}}}\f]
 *
 * The amplitude of the fluid velocity is given by:
 *
 * \f[v_f^2 = \frac{2v^2_{B1}}{1 + \sqrt{1 - 4 v^2_{B0}v^2_{B1}}}\f]
 *
 * The electromagnetic field vectors define a set of basis vectors:
 *
 * \f{align*}{
 * \hat{b}_0 &= \vec{B_0}/B_0 \\
 * \hat{b}_1 &= \vec{B_1}/B_1 \\
 * \hat{e} &= \hat{b}_1 \times \hat{b}_0
 * \f}
 *
 * We also define the auxiliary variable for the phase \f$\phi\f$:
 * \f[\phi = k(\vec{x}\cdot\hat{b}_0 - v_A t)\f]
 * In Cartesian coordinates \f$(x, y, z)\f$, and using
 * dimensionless units, the primitive quantities at a given time \f$t\f$ are
 * then
 *
 * \f{align*}
 * \rho(\vec{x},t) &= \rho_0 \\
 * \vec{v}(\vec{x},t) &= v_f(-\hat{b}_1\cos\phi
 *  +\hat{e}\sin\phi)\\
 * P(\vec{x},t) &= P, \\
 * \epsilon(\vec{x}, t) &= \frac{P}{(\gamma - 1)\rho_0}\\
 * \vec{B}(\vec{x},t) &= B_1(\hat{b}_1\cos\phi
 *  -\hat{e}\sin\phi) + \vec{B_0}
 * \f}
 *
 * Note that the phase speed is not the characteristic Alfv&eacute;n speed
 * \f$c_A\f$, which is the speed in the limiting case where the total magnetic
 * field is parallel to the direction of propagation \cite DelZanna2007pk :
 *
 * \f[c_A^2 = \frac{b^2}{\rho_0 h + b^2}\f]
 *
 * Where \f$b^2\f$ is the invariant quantity \f$B^2 - E^2\f$, given by:
 *
 * \f[b^2 = B_0^2 + B_1^2 - B_0^2 v_f^2\f]
 */
class AlfvenWave {
 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<true>;
  using background_spacetime_type = gr::Solutions::Minkowski<3>;

  /// The wave number of the profile.
  struct WaveNumber {
    using type = double;
    static constexpr OptionString help = {"The wave number of the profile."};
  };

  /// The constant pressure throughout the fluid.
  struct Pressure {
    using type = double;
    static constexpr OptionString help = {
        "The constant pressure throughout the fluid."};
    static type lower_bound() noexcept { return 0.0; }
  };

  /// The constant rest mass density throughout the fluid.
  struct RestMassDensity {
    using type = double;
    static constexpr OptionString help = {
        "The constant rest mass density throughout the fluid."};
    static type lower_bound() noexcept { return 0.0; }
  };

  /// The adiabatic index for the ideal fluid.
  struct AdiabaticIndex {
    using type = double;
    static constexpr OptionString help = {
        "The adiabatic index for the ideal fluid."};
    static type lower_bound() noexcept { return 1.0; }
  };

  /// The background static magnetic field vector.
  struct BackgroundMagneticField {
    using type = std::array<double, 3>;
    static std::string name() noexcept { return "BkgdMagneticField"; }
    static constexpr OptionString help = {
        "The background magnetic field [B0^x, B0^y, B0^z]."};
  };

  /// The sinusoidal magnetic field vector associated with
  /// the Alfv&eacute;n wave, perpendicular to the background
  /// magnetic field vector.
  struct WaveMagneticField {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {
        "The wave magnetic field [B1^x, B1^y, B1^z]."};
  };

  using options =
      tmpl::list<WaveNumber, Pressure, RestMassDensity, AdiabaticIndex,
                 BackgroundMagneticField, WaveMagneticField>;
  static constexpr OptionString help = {
      "Circularly polarized Alfven wave in Minkowski spacetime."};

  AlfvenWave() = default;
  AlfvenWave(const AlfvenWave& /*rhs*/) = delete;
  AlfvenWave& operator=(const AlfvenWave& /*rhs*/) = delete;
  AlfvenWave(AlfvenWave&& /*rhs*/) noexcept = default;
  AlfvenWave& operator=(AlfvenWave&& /*rhs*/) noexcept = default;
  ~AlfvenWave() = default;

  AlfvenWave(WaveNumber::type wavenumber, Pressure::type pressure,
             RestMassDensity::type rest_mass_density,
             AdiabaticIndex::type adiabatic_index,
             BackgroundMagneticField::type background_magnetic_field,
             WaveMagneticField::type wave_magnetic_field) noexcept;

  // @{
  /// Retrieve hydro variable at `(x, t)`
  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x, double t,
      tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x, double t,
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, double /*t*/,
                 tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, double /*t*/,
                 tmpl::list<hydro::Tags::SpatialVelocity<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, double /*t*/,
                 tmpl::list<hydro::Tags::MagneticField<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x, double /*t*/,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x, double /*t*/,
      tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x, double t,
      tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>;
  // @}

  /// Retrieve a collection of hydro variables at `(x, t)`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         double t,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {get<Tags>(variables(x, t, tmpl::list<Tags>{}))...};
  }

  /// Retrieve the metric variables
  template <typename DataType, typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x, double t,
                                     tmpl::list<Tag> /*meta*/) const noexcept {
    return background_spacetime_.variables(x, t, tmpl::list<Tag>{});
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

  const EquationsOfState::IdealFluid<true>& equation_of_state() const noexcept {
    return equation_of_state_;
  }

 private:
  friend bool operator==(const AlfvenWave& lhs, const AlfvenWave& rhs) noexcept;

  // Computes the phase.
  template <typename DataType>
  DataType k_dot_x_minus_vt(const tnsr::I<DataType, 3>& x, double t) const
      noexcept;
  WaveNumber::type wavenumber_ = std::numeric_limits<double>::signaling_NaN();
  Pressure::type pressure_ = std::numeric_limits<double>::signaling_NaN();
  RestMassDensity::type rest_mass_density_ =
      std::numeric_limits<double>::signaling_NaN();
  AdiabaticIndex::type adiabatic_index_ =
      std::numeric_limits<double>::signaling_NaN();
  BackgroundMagneticField::type background_magnetic_field_{
      {std::numeric_limits<double>::signaling_NaN()}};
  WaveMagneticField::type wave_magnetic_field_{
      {std::numeric_limits<double>::signaling_NaN()}};
  EquationsOfState::IdealFluid<true> equation_of_state_{};
  tnsr::I<double, 3, Frame::Inertial>
      initial_unit_vector_along_background_magnetic_field_{};
  tnsr::I<double, 3, Frame::Inertial>
      initial_unit_vector_along_wave_magnetic_field_{};
  tnsr::I<double, 3, Frame::Inertial>
      initial_unit_vector_along_wave_electric_field_{};
  double magnitude_B0_ = std::numeric_limits<double>::signaling_NaN();
  double magnitude_B1_ = std::numeric_limits<double>::signaling_NaN();
  double magnitude_E_ = std::numeric_limits<double>::signaling_NaN();
  double alfven_speed_ = std::numeric_limits<double>::signaling_NaN();
  double fluid_speed_ = std::numeric_limits<double>::signaling_NaN();
  gr::Solutions::Minkowski<3> background_spacetime_{};
};

bool operator!=(const AlfvenWave& lhs, const AlfvenWave& rhs) noexcept;

}  // namespace Solutions
}  // namespace grmhd
