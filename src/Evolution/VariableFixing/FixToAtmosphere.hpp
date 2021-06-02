// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"  // IWYU pragma:  keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace PUP {
class er;
}  // namespace PUP
/// \endcond

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare hydro::Tags::EquationOfStateBase
// IWYU pragma: no_forward_declare hydro::Tags::SpatialVelocity
// IWYU pragma: no_forward_declare hydro::Tags::LorentzFactor
// IWYU pragma: no_forward_declare hydro::Tags::Pressure
// IWYU pragma: no_forward_declare hydro::Tags::RestMassDensity
// IWYU pragma: no_forward_declare hydro::Tags::SpecificEnthalpy
// IWYU pragma: no_forward_declare hydro::Tags::SpecificInternalEnergy

namespace VariableFixing {

/*!
 * \ingroup VariableFixingGroup
 * \brief Fix the primitive variables to an atmosphere in low density regions
 *
 * If the rest mass density is below \f$\rho_{\textrm{cutoff}}\f$
 * (DensityCutoff), it is set to \f$\rho_{\textrm{atm}}\f$
 * (DensityOfAtmosphere), and the pressure, specific internal energy (for
 * one-dimensional equations of state), and specific enthalpy are adjusted to
 * satisfy the equation of state.  For a two-dimensional equation of state, the
 * specific internal energy is set to zero. In addition, the spatial velocity
 * is set to zero, and the Lorentz factor is set to one.
 *
 * If the rest mass density is above \f$\rho_{\textrm{cutoff}}\f$ but below
 * \f$\rho_{\textrm{transition}}\f$ (TransitionDensityCutoff) then the velocity
 * is rescaled such that
 *
 * \f{align*}{
 * \sqrt{v^i v_i}\le \frac{(\rho-\rho_{\textrm{cutoff}})}
 * {(\rho_{\textrm{transition}} - \rho_{\textrm{cutoff}})} v_{\max}
 * \f}
 *
 * where \f$v_{\max}\f$ (MaxVelocityMagnitude) is the maximum allowed magnitude
 * of the velocity. This prescription follows Appendix 2.d of
 * \cite Muhlberger2014pja Note that we require
 * \f$\rho_{\textrm{transition}}\in(\rho_{\textrm{cutoff}},
 *  10\rho_{\textrm{atm}}]\f$
 */
template <size_t Dim>
class FixToAtmosphere {
 public:
  /// \brief Rest mass density of the atmosphere
  struct DensityOfAtmosphere {
    using type = double;
    static type lower_bound() noexcept { return 0.0; }
    static constexpr Options::String help = {"Density of atmosphere"};
  };
  /// \brief Rest mass density at which to impose the atmosphere. Should be
  /// greater than or equal to the density of the atmosphere.
  struct DensityCutoff {
    using type = double;
    static type lower_bound() noexcept { return 0.0; }
    static constexpr Options::String help = {
        "Density to impose atmosphere at. Must be >= rho_atm"};
  };
  /// \brief For densities between DensityOfAtmosphere and
  /// TransitionDensityCutoff the velocity is transitioned away from atmosphere
  /// to avoid abrupt cutoffs.
  ///
  /// This value must not be larger than `10 * DensityOfAtmosphere`.
  struct TransitionDensityCutoff {
    using type = double;
    static type lower_bound() noexcept { return 0.0; }
    static constexpr Options::String help = {
        "For densities between DensityOfAtmosphere and TransitionDensityCutoff "
        "the velocity is transitioned away from atmosphere to avoid abrupt "
        "cutoffs.\n\n"
        "This value must not be larger than 10 * DensityOfAtmosphere."};
  };
  /// \brief The maximum magnitude of the velocity when the density is below
  /// `TransitionDensityCutoff`
  struct MaxVelocityMagnitude {
    using type = double;
    static type lower_bound() noexcept { return 0.0; }
    static type upper_bound() noexcept { return 1.0; }
    static constexpr Options::String help = {
        "The maximum sqrt(v^i v^j gamma_{ij}) allowed when the density is "
        "below TransitionDensityCutoff."};
  };

  using options = tmpl::list<DensityOfAtmosphere, DensityCutoff,
                             TransitionDensityCutoff, MaxVelocityMagnitude>;
  static constexpr Options::String help = {
      "If the rest mass density is below DensityCutoff, it is set\n"
      "to DensityOfAtmosphere, and the pressure, specific internal energy\n"
      "(for one-dimensional equations of state), and specific enthalpy are\n"
      "adjusted to satisfy the equation of state. For a two-dimensional\n"
      "equation of state, the specific internal energy is set to zero.\n"
      "In addition, the spatial velocity is set to zero, and the Lorentz\n"
      "factor is set to one.\n"};

  FixToAtmosphere(double density_of_atmosphere, double density_cutoff,
                  double transition_density_cutoff,
                  double max_velocity_magnitude,
                  const Options::Context& context = {});

  FixToAtmosphere() = default;
  FixToAtmosphere(const FixToAtmosphere& /*rhs*/) = default;
  FixToAtmosphere& operator=(const FixToAtmosphere& /*rhs*/) = default;
  FixToAtmosphere(FixToAtmosphere&& /*rhs*/) noexcept = default;
  FixToAtmosphere& operator=(FixToAtmosphere&& /*rhs*/) noexcept = default;
  ~FixToAtmosphere() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

  using return_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::SpecificEnthalpy<DataVector>>;
  using argument_tags =
      tmpl::list<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataVector>,
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
      gsl::not_null<Scalar<DataVector>*> specific_enthalpy,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& spatial_metric,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
      equation_of_state) const noexcept;

 private:
  template <size_t SpatialDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const FixToAtmosphere<SpatialDim>& lhs,
                         const FixToAtmosphere<SpatialDim>& rhs) noexcept;

  double density_of_atmosphere_{std::numeric_limits<double>::signaling_NaN()};
  double density_cutoff_{std::numeric_limits<double>::signaling_NaN()};
  double transition_density_cutoff_{
      std::numeric_limits<double>::signaling_NaN()};
  double max_velocity_magnitude_{std::numeric_limits<double>::signaling_NaN()};
};

template <size_t Dim>
bool operator!=(const FixToAtmosphere<Dim>& lhs,
                const FixToAtmosphere<Dim>& rhs) noexcept;

}  // namespace VariableFixing
