// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace PUP {
class er;
}  // namespace PUP
namespace hydro {
namespace Tags {
struct EquationOfStateBase;
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct SpatialVelocity;
template <typename DataType>
struct LorentzFactor;
template <typename DataType>
struct Pressure;
template <typename DataType>
struct RestMassDensity;
template <typename DataType>
struct SpecificEnthalpy;
template <typename DataType>
struct SpecificInternalEnergy;
}  // namespace Tags
}  // namespace hydro
/// \endcond

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare Tensor

namespace VariableFixing {

/// \cond
template <size_t ThermodynmaicDim>
class FixToAtmosphere;
/// \endcond

/// \ingroup VariableFixingGroup
/// \brief Fix the primitive variables to an atmosphere in low density regions
///
/// If the rest mass density is below the specified value, it is raised
/// to the specified value, and the pressure, specific internal energy,
/// and specific enthalpy are adjusted to satisfy the equation of state.In
/// addition, the spatial velocity is set to zero, and the Lorentz factor is set
/// to one.
template <>
class FixToAtmosphere<1> {
 public:
  /// \brief Rest mass density of the atmosphere
  struct DensityOfAtmosphere {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr OptionString help = {"Density of atmosphere"};
  };

  using options = tmpl::list<DensityOfAtmosphere>;
  static constexpr OptionString help = {
      "If the rest mass density is below the specified value, it is raised\n"
      "to the specified value, and the pressure, specific internal energy,\n"
      "and specific enthalpy are adjusted to satisfy the equation of state.\n"
      "In addition, the spatial velocity is set to zero, and the Lorentz\n"
      "factor is set to one.\n"};

  explicit FixToAtmosphere(double density_of_atmosphere) noexcept;

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
      const EquationsOfState::EquationOfState<true, 1>& equation_of_state) const
      noexcept;

 private:
  friend bool operator==(const FixToAtmosphere& lhs,
                         const FixToAtmosphere& rhs) noexcept;

  double density_of_atmosphere_{std::numeric_limits<double>::signaling_NaN()};
};

template <size_t ThermodynamicDim>
bool operator!=(const FixToAtmosphere<ThermodynamicDim>& lhs,
                const FixToAtmosphere<ThermodynamicDim>& rhs) noexcept;

}  // namespace VariableFixing
