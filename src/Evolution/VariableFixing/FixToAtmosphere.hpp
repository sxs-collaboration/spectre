// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <limits>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options

namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace VariableFixing {
/// \brief How to apply atmosphere to the reconstructed state.
enum class FixReconstructedStateToAtmosphere : uint8_t {
  Always,
  AtDgFdInterfaceOnly,
  OnFdOnly,
  Never
};

std::ostream& operator<<(std::ostream& os,
                         const FixReconstructedStateToAtmosphere& t);
}  // namespace VariableFixing

template <>
struct Options::create_from_yaml<
    VariableFixing::FixReconstructedStateToAtmosphere> {
  template <typename Metavariables>
  static VariableFixing::FixReconstructedStateToAtmosphere create(
      const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
VariableFixing::FixReconstructedStateToAtmosphere
Options::create_from_yaml<VariableFixing::FixReconstructedStateToAtmosphere>::
    create<void>(const Options::Option& options);

namespace VariableFixing {
/*!
 * \ingroup VariableFixingGroup
 * \brief Fix the primitive variables to an atmosphere in low density regions
 *
 * If the rest mass density is below \f$\rho_{\textrm{cutoff}}\f$
 * (DensityCutoff), it is set to \f$\rho_{\textrm{atm}}\f$
 * (DensityOfAtmosphere), and the pressure, and specific internal energy (for
 * one-dimensional equations of state) are adjusted to satisfy the equation of
 * state.  For a two-dimensional equation of state, the specific internal energy
 * is set to zero.
 */
template <size_t Dim>
class FixToAtmosphere {
  struct Disabled {};
  struct FromEos {};

 public:
  /// \brief Rest mass density of the atmosphere
  struct DensityOfAtmosphere {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {"Density of atmosphere"};
  };
  /// \brief Rest mass density at which to impose the atmosphere. Should be
  /// greater than or equal to the density of the atmosphere.
  struct DensityCutoff {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "Density to impose atmosphere at. Must be >= rho_atm"};
  };

  /*!
   * \brief Limit the velocity in and near the atmosphere.
   *
   * Let $v_{\max}$ be the maximum magnitude of the
   * velocity near the atmosphere, which we typically set to $10^{-4}$.
   * We let $v_{\mathrm{atm}}$ be the maximum magnitude of the velocity
   * in the atmosphere, which we typically set to $0$. We then define
   * the maximum magnitude of the spatial velocity to be
   *
   * \f{align*}{
   *   \tilde{v}
   *   &=\begin{cases}
   *     v_{\mathrm{atm}}, & \mathrm{if}\; \rho < \rho_{v_-} \   \
   *     v_{\mathrm{atm}} + \left(v_{\max} - v_{\mathrm{atm}}\right)
   *     \frac{\rho - \rho_{v_-}}{\rho_{v_+} - \rho_{v_-}},
   *                       & \mathrm{if}\;\rho_{v_-} \le \rho < \rho_{v_+}
   *   \end{cases}
   * \f}
   *
   * We then rescale the velocity by
   *
   * \f{align*}{
   *   v^i\to v^i\frac{\tilde{v}}{\sqrt{v^i\gamma_{ij}v^j}}.
   * \f}
   */
  struct VelocityLimitingOptions {
    struct AtmosphereMaxVelocity {
      using type = double;
      static constexpr Options::String help = {
          "The maximum velocity magnitude IN the atmosphere. Typically set to "
          "0."};
    };

    struct NearAtmosphereMaxVelocity {
      using type = double;
      static constexpr Options::String help = {
          "The maximum velocity magnitude NEAR the atmosphere. Typically set "
          "to 1e-4."};
    };

    struct AtmosphereDensityCutoff {
      using type = double;
      static constexpr Options::String help = {
          "The rest mass density cutoff below which the velocity magnitude is "
          "limited to AtmosphereMaxVelocity. Typically set to "
          "(10 or 20)*DensityOfAtmosphere."};
    };

    struct TransitionDensityBound {
      using type = double;
      static constexpr Options::String help = {
          "The rest mass density above which no velocity limiting is done. "
          "Between "
          "this value and AtmosphereDensityCutoff a linear transition in the "
          "maximum magnitude of the velocity between AtmosphereMaxVelocity and "
          "NearAtmosphereMaxVelocity is done. Typically set to "
          "10*AtmosphereDensityCutoff."};
    };
    using options = tmpl::list<AtmosphereMaxVelocity, NearAtmosphereMaxVelocity,
                               AtmosphereDensityCutoff, TransitionDensityBound>;
    static constexpr Options::String help = {
        "Limit the velocity in and near the atmosphere."};

    // NOLINTNEXTLINE(google-runtime-references)
    void pup(PUP::er& p);

    bool operator==(const VelocityLimitingOptions& rhs) const;
    bool operator!=(const VelocityLimitingOptions& rhs) const;

    double atmosphere_max_velocity{
        std::numeric_limits<double>::signaling_NaN()};
    double near_atmosphere_max_velocity{
        std::numeric_limits<double>::signaling_NaN()};
    double atmosphere_density_cutoff{
        std::numeric_limits<double>::signaling_NaN()};
    double transition_density_bound{
        std::numeric_limits<double>::signaling_NaN()};
  };
  struct VelocityLimiting {
    using type = Options::Auto<VelocityLimitingOptions, Disabled>;
    static constexpr Options::String help = VelocityLimitingOptions::help;
  };

  /*!
   * \brief Options for limiting the temperature in the atmosphere by
   * effectively limiting the polytropic constant, with a generalization for
   * finite temperature equations of state.
   *
   * The basic density limit is fine for a cold EOS, but when a finite
   * temperature EOS is used the temperature needs to be controlled in the
   * atmosphere. While the basic bound of
   * $T\in[T_{\mathrm{min}},T_{\mathrm{max}}]$ is a necessary condition, it is
   * often not sufficient for stability in the atmosphere. We define lower and
   * upper bounds $\rho_{\tau_-}$ and $\rho_{\tau_+}$ for $\rho$ where we apply
   * different limiting behavior when $\rho<\rho_{\tau_-}$,
   * $\rho\in[\rho_{\tau_-},\rho_{\tau_+}]$, and $\rho>\rho_{\tau_+}$.
   *
   * When $\rho<\rho_{\tau_-}$ we set $T=T_{\mathrm{min}}$ if
   * $|T-T_{\mathrm{min}}|<\epsilon_{\kappa_-} |T|$, otherwise we don't change
   * $T$. SpEC uses $\epsilon_{\kappa_-}=10^{-3}$.
   *
   * When $\rho\in[\rho_{\tau_-},\rho_{\tau_+}]$ we apply a more complicated
   * algorithm. We define
   *
   * \f{align*}{
   *   \kappa_{\mathrm{max}} =
   *   1 +
   *   \epsilon_{\kappa,\mathrm{max}}
   *   \min\left(\frac{\rho-\rho_{\tau_-}}{\rho_{\tau_+}-\rho_{\tau_-}},1\right)
   * \f}
   *
   * where $\epsilon_{\kappa,\mathrm{max}}\ge0$ (set to 0.01 in SpEC). With
   * $\kappa_{\mathrm{max}}$ computed we now limit the temperature. We define
   * $p_{\mathrm{min}}=p(\rho,T_{\mathrm{min}},Y_e)$, $p=p(\rho,T,Y_e)$, and
   * $p_{\kappa_{\mathrm{max}}}=p_{\mathrm{min}}\kappa_{\mathrm{max}}$. We then
   * need to find $T$ such that $p(\rho, T, Y_e) = p_{\kappa_{\mathrm{max}}}$.
   * We do this by finding the root of
   *
   * \f{align*}{
   *   f(T')=p(\rho, T', Y_e) - p_{\kappa_{\mathrm{max}}}.
   * \f}
   *
   * We bracket the temperature $T'\in[T_{\min}, T]$, where $T$ is the
   * unmodified temperature. We solve the equation using the TOMS748 algorithm
   * to avoid the need for derivatives of the pressure with respect to the
   * temperature. If $T-T_{\min}<10^{-13}$ we set $T'=\frac{1}{2}(T+T_{\min})$
   * to avoid any root find.
   *
   * When $\rho>\rho_{\tau_+}$ we optionally can apply the same procedure as in
   * the previous case, but with
   * $\kappa_\mathrm{max}=1+\epsilon_{\kappa,\mathrm{max}}$. This is not
   * typically done.
   */
  struct KappaLimitingOptions {
    struct DensityLowerBound {
      using type = double;
      static type lower_bound() { return 0.0; }
      static constexpr Options::String help = {
          "Below this value we set T=T_min if |T-T_min|<EplisonKappaMinus "
          "|T|. Typically set to about a factor of 20 larger than the "
          "atmosphere density."};
    };
    struct EplisonKappaMinus {
      using type = double;
      static constexpr Options::String help = {
          "Used to limit the temperature in conjunction with "
          "DensityLowerBound. See that text for the formula. Typically set "
          "to 1.0e-3."};
    };
    struct DensityUpperBound {
      using type = double;
      static type lower_bound() { return 0.0; }
      static constexpr Options::String help = {
          "The density below which we limit the temperature by limiting the "
          "pressure (at fixed rho, Y_e) to 'p(T)<p_min kappa_max'. Typically "
          "set a factor of 10 larger than DensityLowerBound."};
    };
    struct EpsilonKappaMax {
      using type = double;
      static constexpr Options::String help = {
          "The epsilon factor in the KappaMax computation multiplying the "
          "transition factor. Typically set to 0.01."};
    };
    struct MinTemperature {
      using type = Options::Auto<double, FromEos>;
      static constexpr Options::String help = {"The minimum temperature."};
    };
    struct LimitAboveDensityUpperBound {
      using type = bool;
      static constexpr Options::String help = {
          "If true then we limit the temperature using the KappaMax procedure "
          "at all densities above DensityUpperBound, but with "
          "`KappaMax=1+EplisonKappaMax`. Typically set to False."};
    };
    using options = tmpl::list<DensityLowerBound, EplisonKappaMinus,
                               DensityUpperBound, EpsilonKappaMax,
                               MinTemperature, LimitAboveDensityUpperBound>;
    static constexpr Options::String help = {
        "If set then we apply a limiting precodure on the temperature near the "
        "atmosphere based on essentially limiting the polytropic constant in a "
        "Gamma-law equation of state."};

    // NOLINTNEXTLINE(google-runtime-references)
    void pup(PUP::er& p);

    bool operator==(const KappaLimitingOptions& rhs) const;
    bool operator!=(const KappaLimitingOptions& rhs) const;

    double density_lower_bound{std::numeric_limits<double>::signaling_NaN()};
    double eplison_kappa_minus{std::numeric_limits<double>::signaling_NaN()};
    double density_upper_bound{std::numeric_limits<double>::signaling_NaN()};
    double epsilon_kappa_max{std::numeric_limits<double>::signaling_NaN()};
    std::optional<double> min_temperature{std::nullopt};
    bool limit_above_density_upper_bound{false};
  };
  /// \brief If set then we apply a limiting precodure on the temperature near
  /// the atmosphere based on essentially limiting the polytropic constant in a
  /// Gamma-law equation of state.
  struct KappaLimiting {
    using type = Options::Auto<KappaLimitingOptions, Disabled>;
    static constexpr Options::String help = KappaLimitingOptions::help;
  };

  using options = tmpl::list<DensityOfAtmosphere, DensityCutoff,
                             VelocityLimiting, KappaLimiting>;
  static constexpr Options::String help = {
      "If the rest mass density is below DensityCutoff, it is set\n"
      "to DensityOfAtmosphere, and the pressure, and specific internal energy\n"
      "(for one-dimensional equations of state) are\n"
      "adjusted to satisfy the equation of state. For a two-dimensional\n"
      "equation of state, the specific internal energy is set to zero.\n"
      "In addition, the spatial velocity is set to zero, and the Lorentz\n"
      "factor is set to one.\n"};

  FixToAtmosphere(double density_of_atmosphere, double density_cutoff,
                  std::optional<VelocityLimitingOptions> velocity_limiting,
                  std::optional<KappaLimitingOptions> kappa_limiting,
                  const Options::Context& context = {});

  FixToAtmosphere() = default;
  FixToAtmosphere(const FixToAtmosphere& /*rhs*/) = default;
  FixToAtmosphere& operator=(const FixToAtmosphere& /*rhs*/) = default;
  FixToAtmosphere(FixToAtmosphere&& /*rhs*/) = default;
  FixToAtmosphere& operator=(FixToAtmosphere&& /*rhs*/) = default;
  ~FixToAtmosphere() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  using return_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::Temperature<DataVector>>;
  using argument_tags = tmpl::list<hydro::Tags::ElectronFraction<DataVector>,
                                   gr::Tags::SpatialMetric<DataVector, Dim>,
                                   hydro::Tags::EquationOfStateBase>;

  // for use in `db::mutate_apply`
  template <size_t ThermodynamicDim>
  void operator()(
      gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          spatial_velocity,
      gsl::not_null<Scalar<DataVector>*> lorentz_factor,
      gsl::not_null<Scalar<DataVector>*> pressure,
      gsl::not_null<Scalar<DataVector>*> temperature,
      const Scalar<DataVector>& electron_fraction,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& spatial_metric,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
          equation_of_state) const;

 private:
  template <size_t ThermodynamicDim>
  void set_density_to_atmosphere(
      gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<Scalar<DataVector>*> temperature,
      gsl::not_null<Scalar<DataVector>*> pressure,
      const Scalar<DataVector>& electron_fraction,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
          equation_of_state,
      size_t grid_index) const;

  void apply_velocity_limit(
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          spatial_velocity,
      gsl::not_null<Scalar<DataVector>*> lorentz_factor,
      const Scalar<DataVector>& rest_mass_density,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& spatial_metric,
      size_t grid_index) const;

  template <size_t ThermodynamicDim>
  bool apply_kappa_limit(
      gsl::not_null<Scalar<DataVector>*> temperature,
      const Scalar<DataVector>& rest_mass_density,
      const Scalar<DataVector>& electron_fraction,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
          equation_of_state,
      size_t grid_index) const;

  template <size_t SpatialDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const FixToAtmosphere<SpatialDim>& lhs,
                         const FixToAtmosphere<SpatialDim>& rhs);

  double density_of_atmosphere_{std::numeric_limits<double>::signaling_NaN()};
  double density_cutoff_{std::numeric_limits<double>::signaling_NaN()};
  std::optional<VelocityLimitingOptions> velocity_limiting_{std::nullopt};
  std::optional<KappaLimitingOptions> kappa_limiting_{std::nullopt};
};

template <size_t Dim>
bool operator!=(const FixToAtmosphere<Dim>& lhs,
                const FixToAtmosphere<Dim>& rhs);

}  // namespace VariableFixing
