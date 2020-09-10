// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/KerrSchildCoords.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
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
 * implements the method proposed by \cite Font1998 to initialize the GRMHD
 * variables. The fluid quantities are initialized with their (constant) values
 * far from the black hole, e.g. \f$\rho = \rho_\infty\f$. Here we assume a
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
 * \f$v_\infty^2 = v_i v^i\f$. Finally, following the work by \cite Penner2011,
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
 */
class BondiHoyleAccretion : public MarkAsAnalyticData {
 public:
  using equation_of_state_type = EquationsOfState::PolytropicFluid<true>;

  /// The mass of the black hole, \f$M\f$.
  struct BhMass {
    using type = double;
    static constexpr Options::String help = {"The mass of the black hole."};
    static type lower_bound() noexcept { return 0.0; }
  };
  /// The dimensionless black hole spin, \f$a_* = a/M\f$.
  struct BhDimlessSpin {
    using type = double;
    static constexpr Options::String help = {
        "The dimensionless black hole spin."};
    static type lower_bound() noexcept { return -1.0; }
    static type upper_bound() noexcept { return 1.0; }
  };
  /// The rest mass density of the fluid far from the black hole.
  struct RestMassDensity {
    using type = double;
    static constexpr Options::String help = {
        "The asymptotic rest mass density."};
    static type lower_bound() noexcept { return 0.0; }
  };
  /// The magnitude of the spatial velocity far from the black hole.
  struct FlowSpeed {
    using type = double;
    static constexpr Options::String help = {
        "The magnitude of the asymptotic flow velocity."};
  };
  /// The strength of the magnetic field.
  struct MagFieldStrength {
    using type = double;
    static constexpr Options::String help = {
        "The strength of the magnetic field."};
  };
  /// The polytropic constant of the fluid.
  struct PolytropicConstant {
    using type = double;
    static constexpr Options::String help = {
        "The polytropic constant of the fluid."};
    static type lower_bound() noexcept { return 0.0; }
  };
  /// The polytropic exponent of the fluid.
  struct PolytropicExponent {
    using type = double;
    static constexpr Options::String help = {
        "The polytropic exponent of the fluid."};
    static type lower_bound() noexcept { return 1.0; }
  };

  using options =
      tmpl::list<BhMass, BhDimlessSpin, RestMassDensity, FlowSpeed,
                 MagFieldStrength, PolytropicConstant, PolytropicExponent>;

  static constexpr Options::String help = {
      "Axially symmetric accretion on to a Kerr black hole."};

  BondiHoyleAccretion() = default;
  BondiHoyleAccretion(const BondiHoyleAccretion& /*rhs*/) = delete;
  BondiHoyleAccretion& operator=(const BondiHoyleAccretion& /*rhs*/) = delete;
  BondiHoyleAccretion(BondiHoyleAccretion&& /*rhs*/) noexcept = default;
  BondiHoyleAccretion& operator=(BondiHoyleAccretion&& /*rhs*/) noexcept =
      default;
  ~BondiHoyleAccretion() = default;

  BondiHoyleAccretion(double bh_mass, double bh_dimless_spin,
                      double rest_mass_density, double flow_speed,
                      double magnetic_field_strength,
                      double polytropic_constant,
                      double polytropic_exponent) noexcept;

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
                 tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/)
      const noexcept
      -> tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>;

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
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
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
  tnsr::I<DataType, 3, Frame::NoFrame> spatial_velocity(
      const DataType& r_squared, const DataType& cos_theta,
      const DataType& sin_theta) const noexcept;
  // compute the magnetic field in spherical Kerr-Schild coordinates
  template <typename DataType>
  tnsr::I<DataType, 3, Frame::NoFrame> magnetic_field(
      const DataType& r_squared, const DataType& cos_theta,
      const DataType& sin_theta) const noexcept;

  double bh_mass_ = std::numeric_limits<double>::signaling_NaN();
  double bh_spin_a_ = std::numeric_limits<double>::signaling_NaN();
  double rest_mass_density_ = std::numeric_limits<double>::signaling_NaN();
  double flow_speed_ = std::numeric_limits<double>::signaling_NaN();
  double magnetic_field_strength_ =
      std::numeric_limits<double>::signaling_NaN();
  double polytropic_constant_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent_ = std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::PolytropicFluid<true> equation_of_state_{};
  gr::Solutions::KerrSchild background_spacetime_{};
  gr::KerrSchildCoords kerr_schild_coords_{};
};

bool operator!=(const BondiHoyleAccretion& lhs,
                const BondiHoyleAccretion& rhs) noexcept;

}  // namespace AnalyticData
}  // namespace grmhd
