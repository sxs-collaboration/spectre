// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
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

/// \ingroup VariableFixingGroup
/// \brief Fix the primitive variables to an atmosphere in low density regions
///
/// If the rest mass density is below  \f$\rho_{\textrm{cutoff}}\f$
/// (DensityCutoff), it is set to \f$\rho_{\textrm{atm}}\f$
/// (DensityOfAtmosphere), and the pressure, specific internal energy (for
/// one-dimensional equations of state), and specific enthalpy are adjusted to
/// satisfy the equation of state.  For a two-dimensional equation of state, the
/// specific internal energy is set to zero. In addition, the spatial velocity
/// is set to zero, and the Lorentz factor is set to one.
template <size_t ThermodynamicDim>
class FixToAtmosphere {
 public:
  /// \brief Rest mass density of the atmosphere
  struct DensityOfAtmosphere {
    using type = double;
    static type lower_bound() noexcept { return 0.0; }
    static constexpr OptionString help = {"Density of atmosphere"};
  };
  /// \brief Rest mass density at which to impose the atmosphere. Should be
  /// greater than or equal to the density of the atmosphere.
  struct DensityCutoff {
    using type = double;
    static type lower_bound() noexcept { return 0.0; }
    static constexpr OptionString help = {
        "Density to impose atmosphere at. Must be >= rho_atm"};
  };

  using options = tmpl::list<DensityOfAtmosphere, DensityCutoff>;
  static constexpr OptionString help = {
      "If the rest mass density is below DensityCutoff, it is set\n"
      "to DensityOfAtmosphere, and the pressure, specific internal energy\n"
      "(for one-dimensional equations of state), and specific enthalpy are\n"
      "adjusted to satisfy the equation of state. For a two-dimensional\n"
      "equation of state, the specific internal energy is set to zero.\n"
      "In addition, the spatial velocity is set to zero, and the Lorentz\n"
      "factor is set to one.\n"};

  FixToAtmosphere(double density_of_atmosphere, double density_cutoff,
                  const OptionContext& context = {});

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
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::SpecificEnthalpy<DataVector>>;
  using argument_tags = tmpl::list<hydro::Tags::EquationOfStateBase>;

  void operator()(
      gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> spatial_velocity,
      gsl::not_null<Scalar<DataVector>*> lorentz_factor,
      gsl::not_null<Scalar<DataVector>*> pressure,
      gsl::not_null<Scalar<DataVector>*> specific_enthalpy,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
          equation_of_state) const noexcept;

 private:
  template <size_t LocalThermodynamicDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(
      const FixToAtmosphere<LocalThermodynamicDim>& lhs,
      const FixToAtmosphere<LocalThermodynamicDim>& rhs) noexcept;

  double density_of_atmosphere_{std::numeric_limits<double>::signaling_NaN()};
  double density_cutoff_{std::numeric_limits<double>::signaling_NaN()};
};

template <size_t ThermodynamicDim>
bool operator!=(const FixToAtmosphere<ThermodynamicDim>& lhs,
                const FixToAtmosphere<ThermodynamicDim>& rhs) noexcept;

}  // namespace VariableFixing
