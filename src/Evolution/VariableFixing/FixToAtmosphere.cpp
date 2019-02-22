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

template <size_t ThermodynamicDim>
FixToAtmosphere<ThermodynamicDim>::FixToAtmosphere(
    const double density_of_atmosphere, const double density_cutoff,
    const OptionContext& context)
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
template <size_t ThermodynamicDim>
void FixToAtmosphere<ThermodynamicDim>::pup(PUP::er& p) noexcept {  // NOLINT
  p | density_of_atmosphere_;
  p | density_cutoff_;
}

template <>
void FixToAtmosphere<1>::operator()(
    gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> spatial_velocity,
    gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    gsl::not_null<Scalar<DataVector>*> pressure,
    gsl::not_null<Scalar<DataVector>*> specific_enthalpy,
    const EquationsOfState::EquationOfState<true, 1>& equation_of_state) const
    noexcept {
  for (size_t i = 0; i < rest_mass_density->get().size(); i++) {
    if (UNLIKELY(rest_mass_density->get()[i] < density_cutoff_)) {
      rest_mass_density->get()[i] = density_of_atmosphere_;
      for (size_t d = 0; d < 3; ++d) {
        spatial_velocity->get(d)[i] = 0.0;
      }
      lorentz_factor->get()[i] = 1.0;
      Scalar<double> atmosphere_density{density_of_atmosphere_};
      pressure->get()[i] =
          get(equation_of_state.pressure_from_density(atmosphere_density));
      specific_internal_energy->get()[i] =
          get(equation_of_state.specific_internal_energy_from_density(
              atmosphere_density));
      specific_enthalpy->get()[i] = get(
          equation_of_state.specific_enthalpy_from_density(atmosphere_density));
    }
  }
}

template <>
void FixToAtmosphere<2>::operator()(
    gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> spatial_velocity,
    gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    gsl::not_null<Scalar<DataVector>*> pressure,
    gsl::not_null<Scalar<DataVector>*> specific_enthalpy,
    const EquationsOfState::EquationOfState<true, 2>& equation_of_state) const
    noexcept {
  for (size_t i = 0; i < rest_mass_density->get().size(); i++) {
    if (UNLIKELY(rest_mass_density->get()[i] < density_cutoff_)) {
      rest_mass_density->get()[i] = density_of_atmosphere_;
      specific_internal_energy->get()[i] = 0.0;
      for (size_t d = 0; d < 3; ++d) {
        spatial_velocity->get(d)[i] = 0.0;
      }
      lorentz_factor->get()[i] = 1.0;
      Scalar<double> atmosphere_density{density_of_atmosphere_};
      Scalar<double> atmosphere_energy{0.0};
      pressure->get()[i] =
          get(equation_of_state.pressure_from_density_and_energy(
              atmosphere_density, atmosphere_energy));
      specific_enthalpy->get()[i] =
          get(equation_of_state.specific_enthalpy_from_density_and_energy(
              atmosphere_density, atmosphere_energy));
    }
  }
}

template <size_t LocalThermodynamicDim>
bool operator==(const FixToAtmosphere<LocalThermodynamicDim>& lhs,
                const FixToAtmosphere<LocalThermodynamicDim>& rhs) noexcept {
  return lhs.density_of_atmosphere_ == rhs.density_of_atmosphere_ and
         lhs.density_cutoff_ == rhs.density_cutoff_;
}

template <size_t ThermodynamicDim>
bool operator!=(const FixToAtmosphere<ThermodynamicDim>& lhs,
                const FixToAtmosphere<ThermodynamicDim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                             \
  template class FixToAtmosphere<GET_DIM(data)>;           \
  template bool operator==(                                \
      const FixToAtmosphere<GET_DIM(data)>& lhs,           \
      const FixToAtmosphere<GET_DIM(data)>& rhs) noexcept; \
  template bool operator!=(                                \
      const FixToAtmosphere<GET_DIM(data)>& lhs,           \
      const FixToAtmosphere<GET_DIM(data)>& rhs) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))

#undef GET_DIM
#undef INSTANTIATION

}  // namespace VariableFixing
