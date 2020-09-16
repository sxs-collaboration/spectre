// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/VariableFixing/FixToAtmosphere.hpp"

#include <pup.h>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare Tensor

namespace VariableFixing {

template <size_t Dim, size_t ThermodynamicDim>
FixToAtmosphere<Dim, ThermodynamicDim>::FixToAtmosphere(
    const double density_of_atmosphere, const double density_cutoff,
    const Options::Context& context)
    : density_of_atmosphere_(density_of_atmosphere),
      density_cutoff_(density_cutoff) {
  if (density_of_atmosphere_ > density_cutoff_) {
    PARSE_ERROR(context, "The cutoff density ("
                             << density_cutoff_
                             << ") must be greater than or equal to the "
                                "density value in the atmosphere ("
                             << density_of_atmosphere_ << ')');
  }
}

// clang-tidy: google-runtime-references
template <size_t Dim, size_t ThermodynamicDim>
void FixToAtmosphere<Dim, ThermodynamicDim>::pup(
    PUP::er& p) noexcept {  // NOLINT
  p | density_of_atmosphere_;
  p | density_cutoff_;
}

template <size_t Dim, size_t ThermodynamicDim>
void FixToAtmosphere<Dim, ThermodynamicDim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        spatial_velocity,
    const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<Scalar<DataVector>*> specific_enthalpy,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) const noexcept {
  for (size_t i = 0; i < rest_mass_density->get().size(); i++) {
    if (UNLIKELY(rest_mass_density->get()[i] < density_cutoff_)) {
      rest_mass_density->get()[i] = density_of_atmosphere_;
      for (size_t d = 0; d < Dim; ++d) {
        spatial_velocity->get(d)[i] = 0.0;
      }
      lorentz_factor->get()[i] = 1.0;
      Scalar<double> atmosphere_density{density_of_atmosphere_};
      if constexpr (ThermodynamicDim == 1) {
        pressure->get()[i] =
            get(equation_of_state.pressure_from_density(atmosphere_density));
        specific_internal_energy->get()[i] =
            get(equation_of_state.specific_internal_energy_from_density(
                atmosphere_density));
        specific_enthalpy->get()[i] =
            get(equation_of_state.specific_enthalpy_from_density(
                atmosphere_density));
      } else if constexpr (ThermodynamicDim == 2) {
        Scalar<double> atmosphere_energy{0.0};
        pressure->get()[i] =
            get(equation_of_state.pressure_from_density_and_energy(
                atmosphere_density, atmosphere_energy));
        specific_internal_energy->get()[i] = get(atmosphere_energy);
        specific_enthalpy->get()[i] =
            get(equation_of_state.specific_enthalpy_from_density_and_energy(
                atmosphere_density, atmosphere_energy));
      }
    }
  }
}

template <size_t Dim, size_t LocalThermodynamicDim>
bool operator==(
    const FixToAtmosphere<Dim, LocalThermodynamicDim>& lhs,
    const FixToAtmosphere<Dim, LocalThermodynamicDim>& rhs) noexcept {
  return lhs.density_of_atmosphere_ == rhs.density_of_atmosphere_ and
         lhs.density_cutoff_ == rhs.density_cutoff_;
}

template <size_t Dim, size_t ThermodynamicDim>
bool operator!=(const FixToAtmosphere<Dim, ThermodynamicDim>& lhs,
                const FixToAtmosphere<Dim, ThermodynamicDim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(r, data)                                           \
  template class FixToAtmosphere<DIM(data), THERMO_DIM(data)>;           \
  template bool operator==(                                              \
      const FixToAtmosphere<DIM(data), THERMO_DIM(data)>& lhs,           \
      const FixToAtmosphere<DIM(data), THERMO_DIM(data)>& rhs) noexcept; \
  template bool operator!=(                                              \
      const FixToAtmosphere<DIM(data), THERMO_DIM(data)>& lhs,           \
      const FixToAtmosphere<DIM(data), THERMO_DIM(data)>& rhs) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2))

#undef DIM
#undef THERMO_DIM
#undef INSTANTIATION

}  // namespace VariableFixing
