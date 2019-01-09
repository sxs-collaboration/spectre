// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace grmhd {
namespace Solutions {

/*!
 * \brief Bondi-Michel accretion with superposed magnetic field in Schwarzschild
 * spacetime in Kerr-Schild coordinates.
 *
 * An analytic solution to the 3-D GrMhd system. The user specifies the sonic
 * radius \f$r_c\f$ and sonic rest mass density \f$\rho_c\f$, which are the
 * radius and rest mass density at the sonic point, the radius at which the
 * fluid's Eulerian velocity as seen by a distant observer overtakes the local
 * sound speed \f$c_{s,c}\f$. With a specified polytropic exponent \f$\Gamma\f$,
 * these quantities can be related to the sound speed at infinity
 * \f$c_{s,\inf}\f$ using the following relations:
 *
 * \f{align*}
 * c_{s,c}^2 &= \frac{1}{2r_c - 3} \\
 * c_{s,\inf}^2 &= \Gamma - 1 + (c_{s,c}^2 - \Gamma + 1)\sqrt{1 + 3c_{s,c}^2}
 * \f}
 *
 * In the case of the interstellar medium, the sound
 * speed is \f$\approx 10^{-4}\f$, which results in a sonic radius of
 * \f$\approx 10^8 M\f$ for \f$\Gamma \neq 5/3\f$ (Rezzola and Zanotti, 2013).
 * However, for numerical-testing purposes, it is more common to use a value of
 * \f$\approx 10 M\f$.
 *
 * The density is found via root-finding, through the
 * Bernoulli equation. As one approaches the sonic radius, a second root makes
 * an appearance and one must take care to bracket the correct root. This is
 * done by using the upper bound \f$\frac{\dot{M}}{4\pi}\sqrt{\frac{2}{Mr^3}}\f$
 *
 * Additionally specified by the user are the polytropic exponent \f$\Gamma\f$,
 * and the strength parameter of the magnetic field \f$B\f$.
 * In Kerr-Schild-Cartesian coordinates \f$(x, y, z)\f$, where
 * \f$ r = \sqrt{x^2 + y^2 + z^2}\f$, the superposed magnetic field is
 * \f{align*}
 * B_x(\vec{x},t) &= \frac{B x}{r^3 \sqrt{1 + 2/r}} \\
 * B_y(\vec{x},t) &= \frac{B y}{r^3 \sqrt{1 + 2/r}} \\
 * B_z(\vec{x},t) &= \frac{B z}{r^3 \sqrt{1 + 2/r}}
 * \f}
 */
class BondiMichel {
  template <typename DataType>
  struct IntermediateVars;

 public:
  using equation_of_state_type = EquationsOfState::PolytropicFluid<true>;
  using background_spacetime_type = gr::Solutions::KerrSchild;

  /// The mass of the black hole.
  struct Mass {
    using type = double;
    static constexpr OptionString help = {"Mass of the black hole."};
    static type lower_bound() noexcept { return 0.0; }
  };

  /// The radius at which the fluid becomes supersonic.
  struct SonicRadius {
    using type = double;
    static constexpr OptionString help = {
        "Schwarzschild radius where fluid becomes supersonic."};
    static type lower_bound() noexcept { return 0.0; }
  };

  /// The rest mass density of the fluid at the sonic radius.
  struct SonicDensity {
    using type = double;
    static constexpr OptionString help = {
        "The density of the fluid at the sonic radius."};
    static type lower_bound() noexcept { return 0.0; }
  };

  /// The polytropic exponent for the polytropic fluid.
  struct PolytropicExponent {
    using type = double;
    static constexpr OptionString help = {
        "The polytropic exponent for the polytropic fluid."};
    static type lower_bound() noexcept { return 1.0; }
  };

  /// The strength of the radial magnetic field.
  struct MagFieldStrength {
    using type = double;
    static constexpr OptionString help = {
        "The strength of the radial magnetic field."};
  };

  using options = tmpl::list<Mass, SonicRadius, SonicDensity,
                             PolytropicExponent, MagFieldStrength>;
  static constexpr OptionString help = {
      "Bondi-Michel solution with a radial magnetic field using \n"
      "the Schwarzschild coordinate system. Quantities prefixed with \n"
      "`sonic` refer to field quantities evaluated at the radius \n"
      "where the fluid speed overtakes the sound speed."};

  BondiMichel() = default;
  BondiMichel(const BondiMichel& /*rhs*/) = delete;
  BondiMichel& operator=(const BondiMichel& /*rhs*/) = delete;
  BondiMichel(BondiMichel&& /*rhs*/) noexcept = default;
  BondiMichel& operator=(BondiMichel&& /*rhs*/) noexcept = default;
  ~BondiMichel() = default;

  BondiMichel(Mass::type mass, SonicRadius::type sonic_radius,
              SonicDensity::type sonic_density,
              PolytropicExponent::type polytropic_exponent,
              MagFieldStrength::type mag_field_strength) noexcept;

  /// Retrieve a collection of  hydro variables at `(x, t)`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         const double /*t*/,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    // non-const so we can move out the metric vars. We assume that no variable
    // is being retrieved more than once, which would cause problems with
    // TaggedTuple anyway.
    auto intermediate_vars = IntermediateVars<DataType>{
        rest_mass_density_at_infinity_,
        mass_accretion_rate_over_four_pi_,
        mass_,
        polytropic_constant_,
        polytropic_exponent_,
        bernoulli_constant_squared_minus_one_,
        sonic_radius_,
        sonic_density_,
        x,
        tmpl2::flat_any_v<
            not tmpl::list_contains_v<hydro::grmhd_tags<DataType>, Tags>...>,
        background_spacetime_};
    return {get<Tags>(variables(x, tmpl::list<Tags>{}, intermediate_vars))...};
  }

  template <typename DataType, typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x,
                                     const double /*t*/,  // NOLINT
                                     tmpl::list<Tag> /*meta*/) const noexcept {
    return variables(
        x, tmpl::list<Tag>{},
        IntermediateVars<DataType>{
            rest_mass_density_at_infinity_, mass_accretion_rate_over_four_pi_,
            mass_, polytropic_constant_, polytropic_exponent_,
            bernoulli_constant_squared_minus_one_, sonic_radius_,
            sonic_density_, x,
            not tmpl::list_contains_v<hydro::grmhd_tags<DataType>, Tag>,
            background_spacetime_});
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT
  const EquationsOfState::PolytropicFluid<true>& equation_of_state() const
      noexcept {
    return equation_of_state_;
  }

 private:
  friend bool operator==(const BondiMichel& lhs,
                         const BondiMichel& rhs) noexcept;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/,
                 const IntermediateVars<DataType>& vars) const noexcept
      -> tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/,
      const IntermediateVars<DataType>& vars) const noexcept
      -> tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/,
                 const IntermediateVars<DataType>& vars) const noexcept
      -> tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<
          hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>> /*meta*/,
      const IntermediateVars<DataType>& vars) const noexcept
      -> tuples::TaggedTuple<
          hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<
          hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>> /*meta*/,
      const IntermediateVars<DataType>& vars) const noexcept
      -> tuples::TaggedTuple<
          hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/,
      const IntermediateVars<DataType>& vars) const noexcept
      -> tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/,
                 const IntermediateVars<DataType>& vars) const noexcept
      -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/,
                 const IntermediateVars<DataType>& vars) const noexcept
      -> tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>;

  template <typename DataType, typename Tag,
            Requires<not tmpl::list_contains_v<hydro::grmhd_tags<DataType>,
                                               Tag>> = nullptr>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& /*x*/,
                                     tmpl::list<Tag> /*meta*/,
                                     IntermediateVars<DataType>& vars) const
      noexcept {
    return {std::move(get<Tag>(vars.kerr_schild_soln))};
  }

  template <typename DataType>
  struct IntermediateVars {
    IntermediateVars(
        double rest_mass_density_at_infinity,
        double in_mass_accretion_rate_over_four_pi, double in_mass,
        double in_polytropic_constant, double in_polytropic_exponent,
        double in_bernoulli_constant_squared_minus_one, double in_sonic_radius,
        double in_sonic_density, const tnsr::I<DataType, 3>& x,
        bool need_spacetime,
        const gr::Solutions::KerrSchild& background_spacetime) noexcept;
    DataType radius{};
    DataType rest_mass_density{};
    double mass_accretion_rate_over_four_pi{};
    double mass{};
    double polytropic_constant{};
    double polytropic_exponent{};
    double bernoulli_constant_squared_minus_one{};
    double sonic_radius{};
    double sonic_density{};
    double bernoulli_root_function(double rest_mass_density_guess,
                                   double current_radius) const noexcept;
    tuples::tagged_tuple_from_typelist<
        typename gr::Solutions::KerrSchild::tags<DataType>>
        kerr_schild_soln{};
  };

  Mass::type mass_ = std::numeric_limits<double>::signaling_NaN();
  SonicRadius::type sonic_radius_ =
      std::numeric_limits<double>::signaling_NaN();
  SonicDensity::type sonic_density_ =
      std::numeric_limits<double>::signaling_NaN();
  PolytropicExponent::type polytropic_exponent_ =
      std::numeric_limits<double>::signaling_NaN();
  MagFieldStrength::type mag_field_strength_ =
      std::numeric_limits<double>::signaling_NaN();
  double sonic_fluid_speed_squared_ =
      std::numeric_limits<double>::signaling_NaN();
  double sonic_sound_speed_squared_ =
      std::numeric_limits<double>::signaling_NaN();
  double polytropic_constant_ = std::numeric_limits<double>::signaling_NaN();
  double mass_accretion_rate_over_four_pi_ =
      std::numeric_limits<double>::signaling_NaN();
  double bernoulli_constant_squared_minus_one_ =
      std::numeric_limits<double>::signaling_NaN();
  double rest_mass_density_at_infinity_ =
      std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::PolytropicFluid<true> equation_of_state_{};
  gr::Solutions::KerrSchild background_spacetime_{};
};

bool operator!=(const BondiMichel& lhs, const BondiMichel& rhs) noexcept;

}  // namespace Solutions
}  // namespace grmhd
