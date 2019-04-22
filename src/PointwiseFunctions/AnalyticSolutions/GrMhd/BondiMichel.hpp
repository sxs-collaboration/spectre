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
 * \brief Bondi-Michel accretion \cite Michel1972 with superposed magnetic field
 * in Schwarzschild spacetime in Cartesian Kerr-Schild coordinates.
 *
 * An analytic solution to the 3-D GRMHD system. The user specifies the sonic
 * radius \f$r_c\f$ and sonic rest mass density \f$\rho_c\f$, which are the
 * radius and rest mass density at the sonic point, the radius at which the
 * fluid's Eulerian velocity as seen by a distant observer overtakes the local
 * sound speed \f$c_{s,c}\f$. With a specified polytropic exponent \f$\gamma\f$,
 * these quantities can be related to the sound speed at infinity
 * \f$c_{s,\infty}\f$ using the following relations:
 *
 * \f{align*}
 * c_{s,c}^2 &= \frac{M}{2r_c - 3M} \\
 * c_{s,\infty}^2 &= \gamma - 1 + (c_{s,c}^2 - \gamma + 1)\sqrt{1 + 3c_{s,c}^2}
 * \f}
 *
 * In the case of the interstellar medium, the sound
 * speed is \f$\approx 10^{-4}\f$, which results in a sonic radius of
 * \f$\approx 10^8 M\f$ for \f$\gamma \neq 5/3\f$ \cite RezzollaBook.
 *
 * The density is found via root-finding, through the
 * Bernoulli equation. As one approaches the sonic radius, a second root makes
 * an appearance and one must take care to bracket the correct root. This is
 * done by using the upper bound
 * \f$\frac{\dot{M}}{4\pi}\sqrt{\frac{2}{Mr^3}}\f$.
 *
 * Additionally specified by the user are the polytropic exponent \f$\gamma\f$,
 * and the strength parameter of the magnetic field \f$B_0\f$.
 * In Cartesian Kerr-Schild coordinates \f$(x, y, z)\f$, where
 * \f$ r = \sqrt{x^2 + y^2 + z^2}\f$, the superposed magnetic field is
 * \cite Etienne2010ui
 *
 * \f{align*}
 * B^i(\vec{x},t) = \frac{B_0 M^2}{r^3 \sqrt{\gamma}}x^i
 *  =\frac{B_0 M^2}{r^3 \sqrt{1 + 2M/r}}x^i.
 * \f}
 *
 * The accretion rate is
 *
 * \f{align*}{
 * \dot{M}=4\pi r^2\rho u^r,
 * \f}
 *
 * and at the sonic radius
 *
 * \f{align*}{
 * \dot{M}_c=\sqrt{8}\pi \sqrt{M}r_c^{3/2}\rho_c.
 * \f}
 *
 * The polytropic constant is given by
 *
 * \f{align*}{
 * K=\frac{1}{\gamma\rho_c^{\gamma-1}}
 * \left[\frac{M(\gamma-1)}{(2r_c-3M)(\gamma-1)-M}\right].
 * \f}
 *
 * The density as a function of the sound speed is
 *
 * \f{align*}{
 * \rho^{\gamma-1}=\frac{(\gamma-1)c_s^2}{\gamma K(\gamma-1-c_s^2)}.
 * \f}
 *
 * #### Horizon quantities, \f$\gamma\ne5/3\f$
 * The density at the horizon is given by:
 *
 * \f{align*}{
 *  \rho_h\simeq\frac{1}{16}
 *  \left(\frac{5-3\gamma}{2}\right)^{(3\gamma-5)/[2(\gamma-1)]}
 *  \frac{\rho_\infty}{c_{s,\infty}^3}.
 * \f}
 *
 * Using the Lorentz invariance of \f$b^2\f$ we evaluate:
 *
 * \f{align*}{
 * b^2=\frac{B^2}{W^2}+(B^i v_i)^2=
 *  B^r B^r(1-\gamma_{rr}v^r v^r)+B^r B^r v^r v^r
 * =B^r B^r = \frac{B_0^2 M^4}{r^4},
 * \f}
 *
 * where \f$r\f$ is the Cartesian Kerr-Schild radius, which is equal to the
 * areal radius for a non-spinning black hole. At the horizon we get
 *
 * \f{align*}{
 * b^2_h=\frac{B^2_0}{16}.
 * \f}
 *
 * Finally, we get
 *
 * \f{align*}{
 * B_0 = 4 \sqrt{b^2_h} = 4\sqrt{\rho_h} \sqrt{\frac{b^2_h}{\rho_h}},
 * \f}
 *
 * where the last equality is useful for comparison to papers that give
 * \f$b^2_h/\rho_h\f$.
 *
 * To help with comparing to other codes the following script can be used to
 * compute \f$b^2_h/\rho_h\f$:
 *
 * \code{.py}
 * #!/bin/env python
 *
 * import numpy as np
 *
 * # Input parameters
 * B_0 = 18
 * r_c = 8
 * rho_c = 1 / 16
 * gamma = 4 / 3
 * mass = 1
 *
 * K = 1 / (gamma * rho_c**(gamma - 1)) * ((gamma - 1) * mass) / (
 *     (2 * r_c - 3 * mass) * (gamma - 1) - mass)
 * c_s_c = mass / (2 * r_c - 3 * mass)
 * c_inf = gamma - 1 + (c_s_c - gamma + 1) * np.sqrt(1. + 3. * c_s_c)
 * rho_inf = ((gamma - 1) * c_inf / (gamma * K *
 *                                   (gamma - 1 - c_inf)))**(1. / (gamma - 1.))
 * rho_h = 1. / 16. * (2.5 - 1.5 * gamma)**(
 *     (3 * gamma - 5) / (2 * (gamma - 1))) * rho_inf / (c_inf**1.5)
 *
 * print("B_0", B_0)
 * print("r_c: ", r_c)
 * print("rho_c", rho_c)
 * print("b_h^2/rho_h: ", B_0**2 / (16. * rho_h))
 * print("gamma: ", gamma)
 * \endcode
 *
 * #### Horizon quantities, \f$\gamma=5/3\f$
 * The density at the horizon is given by:
 *
 * \f{align*}{
 * \rho_h\simeq \frac{1}{16}\frac{\rho_\infty}{u_h c_{s,\infty}^3},
 * \f}
 *
 * which gives \cite RezzollaBook
 *
 * \f{align*}{
 * \rho_h\simeq 0.08\frac{\rho_\infty}{c_{s,\infty}^3}.
 * \f}
 *
 * The magnetic field \f$b^2\f$ is the same as the \f$\gamma\ne5/3\f$.
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
