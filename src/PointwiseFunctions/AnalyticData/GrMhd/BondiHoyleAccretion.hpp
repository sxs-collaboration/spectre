// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/KerrSchildCoords.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma:  no_include <pup.h>

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
/// \endcond

namespace grmhd {
namespace AnalyticData {

/*!
 * \brief Analytic initial data for axially symmetric Bondi-Hoyle accretion.
 *
 * In the context of studying Bondi-Hoyle accretion, i.e. non-spherical
 * accretion on to a Kerr black hole moving relative to a gas cloud, this class
 * implements the method proposed by Font & Ibáñez (1998) \ref font_98 "[1]" to
 * initialize the GRMHD variables. The fluid quantities are initialized with
 * their (constant) values far from the black hole, e.g.
 * \f$\rho = \rho_\infty\f$. Here we assume a
 * polytropic equation of state, so only the rest mass density, as well as the
 * polytropic constant and the polytropic exponent, are provided as inputs.
 * The spatial velocity is initialized using a field that ensures that the
 * injected gas reproduces a continuous parallel wind at large distances.
 * The direction of this flow is chosen to be along the black hole spin.
 * In Kerr (or "spherical Kerr-Schild", see gr::KerrSchildCoords) coordinates,
 *
 * \f{align*}
 * v^r &= \frac{1}{\sqrt{\gamma_{rr}}}v_\infty \cos\theta\\
 * v^\theta &= -\frac{1}{\sqrt{\gamma_{\theta\theta}}}v_\infty \sin\theta\\
 * v^\phi &= 0.
 * \f}
 *
 * where \f$\gamma_{ij} = g_{ij}\f$ is the spatial metric, and \f$v_\infty\f$
 * is the flow speed far from the black hole. Note that
 * \f$v_\infty^2 = v_i v^i\f$. Finally, following the work by
 * Penner (2011) \ref penner_11 "[2]",
 * the magnetic field is initialized using Wald's solution to Maxwell's
 * equations in Kerr black hole spacetime. In Kerr ("spherical
 * Kerr-Schild") coordinates, the spatial components of the Faraday tensor read
 *
 * \f{align*}
 * F_{r\theta} &= a B_0 \left[1 + \frac{2Mr}{\Sigma^2}(r^2 - a^2)\right]
 * \sin\theta\cos\theta\\
 * F_{\theta\phi} &= B_0\left[\Delta +
 * \frac{2Mr}{\Sigma^2}(r^4 - a^4)\right]\sin\theta\cos\theta\\
 * F_{\phi r} &= - B_0\left[r + \frac{M a^2}{\Sigma^2}
 * (r^2 - a^2\cos^2\theta)(1 + \cos^2\theta)\right]\sin^2\theta.
 * \f}
 *
 * where \f$\Sigma = r^2 + a^2\cos^2\theta\f$ and \f$\Delta = r^2 - 2Mr +
 * a^2\f$. The associated Eulerian magnetic field is
 *
 * \f{align*}
 * B^r = \frac{F_{\theta\phi}}{\sqrt\gamma},\quad
 * B^\theta = \frac{F_{\phi r}}{\sqrt\gamma},\quad
 * B^\phi = \frac{F_{r\theta}}{\sqrt\gamma}.
 * \f}
 *
 * where \f$\gamma = \text{det}(\gamma_{ij})\f$. Wald's solution reproduces a
 * uniform magnetic field far from the black hole.
 *
 * \anchor font_98 [1] J.A. Font & J.M. Ibáñez, ApJ
 * [494 (1998) 297](http://esoads.eso.org/abs/1998ApJ...494..297F)
 *
 * \anchor penner_11 [2] A.J. Penner, MNRAS
 * [414 (2011) 1467](http://cdsads.u-strasbg.fr/abs/2011MNRAS.414.1467P)
 */
class BondiHoyleAccretion {
  using equation_of_state_type = EquationsOfState::PolytropicFluid<true>;
  using background_spacetime_type = gr::Solutions::KerrSchild;

 public:
  /// The mass of the black hole, \f$M\f$.
  struct BhMass {
    using type = double;
    static constexpr OptionString help = {"The mass of the black hole."};
    static type lower_bound() noexcept { return 0.0; }
  };
  /// The dimensionless black hole spin, \f$a_* = a/M\f$.
  struct BhDimlessSpin {
    using type = double;
    static constexpr OptionString help = {"The dimensionless black hole spin."};
    static type lower_bound() noexcept { return -1.0; }
    static type upper_bound() noexcept { return 1.0; }
  };
  /// The rest mass density of the fluid far from the black hole.
  struct RestMassDensity {
    using type = double;
    static constexpr OptionString help = {"The asymptotic rest mass density."};
    static type lower_bound() noexcept { return 0.0; }
  };
  /// The magnitude of the spatial velocity far from the black hole.
  struct FlowSpeed {
    using type = double;
    static constexpr OptionString help = {
        "The magnitude of the asymptotic flow velocity."};
  };
  /// The strength of the magnetic field.
  struct MagFieldStrength {
    using type = double;
    static constexpr OptionString help = {
        "The strength of the magnetic field."};
  };
  /// The polytropic constant of the fluid.
  struct PolytropicConstant {
    using type = double;
    static constexpr OptionString help = {
        "The polytropic constant of the fluid."};
    static type lower_bound() noexcept { return 0.0; }
  };
  /// The polytropic exponent of the fluid.
  struct PolytropicExponent {
    using type = double;
    static constexpr OptionString help = {
        "The polytropic exponent of the fluid."};
    static type lower_bound() noexcept { return 1.0; }
  };

  using options =
      tmpl::list<BhMass, BhDimlessSpin, RestMassDensity, FlowSpeed,
                 MagFieldStrength, PolytropicConstant, PolytropicExponent>;

  static constexpr OptionString help = {
      "Axially symmetric accretion on to a Kerr black hole."};

  BondiHoyleAccretion() = default;
  BondiHoyleAccretion(const BondiHoyleAccretion& /*rhs*/) = delete;
  BondiHoyleAccretion& operator=(const BondiHoyleAccretion& /*rhs*/) = delete;
  BondiHoyleAccretion(BondiHoyleAccretion&& /*rhs*/) noexcept = default;
  BondiHoyleAccretion& operator=(BondiHoyleAccretion&& /*rhs*/) noexcept =
      default;
  ~BondiHoyleAccretion() = default;

  BondiHoyleAccretion(BhMass::type bh_mass, BhDimlessSpin::type bh_dimless_spin,
                      RestMassDensity::type rest_mass_density,
                      FlowSpeed::type flow_speed,
                      MagFieldStrength::type magnetic_field_strength,
                      PolytropicConstant::type polytropic_constant,
                      PolytropicExponent::type polytropic_exponent) noexcept;

  // @{
  /// Retrieve hydro variable at `x`
  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::SpatialVelocity<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::MagneticField<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>;
  // @}

  /// Retrieve a collection of hydro variables at `x`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  /// Retrieve the metric variables at `x`
  template <typename DataType, typename Tag,
            Requires<not tmpl::list_contains_v<hydro::grmhd_tags<DataType>,
                                               Tag>> = nullptr>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x,
                                     tmpl::list<Tag> /*meta*/) const noexcept {
    constexpr double dummy_time = 0.0;
    return {std::move(get<Tag>(background_spacetime_.variables(
        x, dummy_time, gr::Solutions::KerrSchild::tags<DataType>{})))};
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

  const EquationsOfState::PolytropicFluid<true>& equation_of_state() const
      noexcept {
    return equation_of_state_;
  }

 private:
  friend bool operator==(const BondiHoyleAccretion& lhs,
                         const BondiHoyleAccretion& rhs) noexcept;

  // compute the spatial velocity in spherical Kerr-Schild coordinates
  template <typename DataType>
  typename hydro::Tags::SpatialVelocity<DataType, 3, Frame::NoFrame>::type
  spatial_velocity(const DataType& r_squared, const DataType& cos_theta,
                   const DataType& sin_theta) const noexcept;
  // compute the magnetic field in spherical Kerr-Schild coordinates
  template <typename DataType>
  typename hydro::Tags::MagneticField<DataType, 3, Frame::NoFrame>::type
  magnetic_field(const DataType& r_squared, const DataType& cos_theta,
                 const DataType& sin_theta) const noexcept;

  BhMass::type bh_mass_ = std::numeric_limits<double>::signaling_NaN();
  BhDimlessSpin::type bh_spin_a_ = std::numeric_limits<double>::signaling_NaN();
  RestMassDensity::type rest_mass_density_ =
      std::numeric_limits<double>::signaling_NaN();
  FlowSpeed::type flow_speed_ = std::numeric_limits<double>::signaling_NaN();
  MagFieldStrength::type magnetic_field_strength_ =
      std::numeric_limits<double>::signaling_NaN();
  PolytropicConstant::type polytropic_constant_ =
      std::numeric_limits<double>::signaling_NaN();
  PolytropicExponent::type polytropic_exponent_ =
      std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::PolytropicFluid<true> equation_of_state_{};
  gr::Solutions::KerrSchild background_spacetime_{};
  gr::KerrSchildCoords kerr_schild_coords_{};
};

bool operator!=(const BondiHoyleAccretion& lhs,
                const BondiHoyleAccretion& rhs) noexcept;

}  // namespace AnalyticData
}  // namespace grmhd
