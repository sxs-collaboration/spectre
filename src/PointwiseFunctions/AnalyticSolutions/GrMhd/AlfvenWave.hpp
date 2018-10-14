// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

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
 * spacetime travelling in the z-direction.
 *
 * An analytic solution to the 3-D GrMhd system. The user specifies the
 * wavenumber \f$k_z\f$ of the Alfv&eacute;n wave, the constant pressure
 * throughout the fluid \f$P\f$, the constant rest mass density throughout the
 * fluid \f$\rho_0\f$, the adiabatic index for the ideal fluid equation of
 * state \f$\gamma\f$, the background magnetic field strength \f$B_0\f$,
 * and the strength of the perturbation of the magnetic field \f$B_1\f$.
 * The Alfv&eacute;n wave phase speed is then given by:
 *
 * \f[v_A = \frac{B_0}{\sqrt{\rho_0 h + B_0^2}}\f]
 *
 * The amplitude of the fluid velocity is given by:
 *
 * \f[v_f = \frac{-B_1}{\sqrt{\rho_0 h + B_0^2}}\f]
 *
 * In Cartesian coordinates \f$(x, y, z)\f$, and using
 * dimensionless units, the primitive quantities at a given time \f$t\f$ are
 * then
 *
 * \f{align*}
 * \rho(\vec{x},t) &= \rho_0 \\
 * v_x(\vec{x},t) &= v_f \cos(k_z(z - v_A t))\\
 * v_y(\vec{x},t) &= v_f \sin(k_z(z - v_A t))\\
 * v_z(\vec{x},t) &= 0\\
 * P(\vec{x},t) &= P, \\
 * \epsilon(\vec{x}, t) &= \frac{P}{(\gamma - 1)\rho_0}\\
 * B_x(\vec{x},t) &= B_1 \cos(k_z(z - v_A t))\\
 * B_y(\vec{x},t) &= B_1 \sin(k_z(z - v_A t))\\
 * B_z(\vec{x},t) &= B_0
 * \f}
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
    static type lower_bound() { return 0.0; }
  };

  /// The constant rest mass density throughout the fluid.
  struct RestMassDensity {
    using type = double;
    static constexpr OptionString help = {
        "The constant rest mass density throughout the fluid."};
    static type lower_bound() { return 0.0; }
  };

  /// The adiabatic index for the ideal fluid.
  struct AdiabaticIndex {
    using type = double;
    static constexpr OptionString help = {
        "The adiabatic index for the ideal fluid."};
    static type lower_bound() { return 1.0; }
  };

  /// The strength of the background magnetic field.
  struct BackgroundMagField {
    using type = double;
    static constexpr OptionString help = {
        "The background magnetic field strength."};
  };

  /// The amplitude of the perturbation of the magnetic field.
  struct PerturbationSize {
    using type = double;
    static constexpr OptionString help = {
        "The perturbation amplitude of the magnetic field."};
    static type lower_bound() { return -1.0; }
    static type upper_bound() { return 1.0; }
  };

  using options =
      tmpl::list<WaveNumber, Pressure, RestMassDensity, AdiabaticIndex,
                 BackgroundMagField, PerturbationSize>;
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
             BackgroundMagField::type background_mag_field,
             PerturbationSize::type perturbation_size) noexcept;

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

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT
  WaveNumber::type wavenumber() const noexcept { return wavenumber_; }
  Pressure::type pressure() const noexcept { return pressure_; }
  RestMassDensity::type rest_mass_density() const noexcept {
    return rest_mass_density_;
  }
  AdiabaticIndex::type adiabatic_index() const noexcept {
    return adiabatic_index_;
  }
  BackgroundMagField::type background_mag_field() const noexcept {
    return background_mag_field_;
  }
  PerturbationSize::type perturbation_size() const noexcept {
    return perturbation_size_;
  }
  double alfven_speed() const noexcept { return alfven_speed_; }
  double fluid_speed() const noexcept { return fluid_speed_; }

  const EquationsOfState::IdealFluid<true>& equation_of_state() const noexcept {
    return equation_of_state_;
  }

  const gr::Solutions::Minkowski<3>& background_spacetime() const noexcept {
    return background_spacetime_;
  }

 private:
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
  BackgroundMagField::type background_mag_field_ =
      std::numeric_limits<double>::signaling_NaN();
  PerturbationSize::type perturbation_size_ =
      std::numeric_limits<double>::signaling_NaN();
  double alfven_speed_ = std::numeric_limits<double>::signaling_NaN();
  double fluid_speed_ = std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::IdealFluid<true> equation_of_state_{};
  gr::Solutions::Minkowski<3> background_spacetime_{};
};

bool operator==(const AlfvenWave& lhs, const AlfvenWave& rhs) noexcept;

bool operator!=(const AlfvenWave& lhs, const AlfvenWave& rhs) noexcept;

}  // namespace Solutions
}  // namespace grmhd
