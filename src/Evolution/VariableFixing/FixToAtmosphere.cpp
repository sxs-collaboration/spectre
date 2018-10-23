// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/VariableFixing/FixToAtmosphere.hpp"

#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"

// IWYU pragma: no_include <array>
// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare Tensor

namespace VariableFixing {

FixToAtmosphere<1>::FixToAtmosphere(const double density_of_atmosphere) noexcept
    : density_of_atmosphere_(density_of_atmosphere) {}

// clang-tidy: google-runtime-references
void FixToAtmosphere<1>::pup(PUP::er& p) noexcept {  // NOLINT
  p | density_of_atmosphere_;
}

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
    if (UNLIKELY(rest_mass_density->get()[i] < density_of_atmosphere_)) {
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

bool operator==(const FixToAtmosphere<1>& lhs,
                const FixToAtmosphere<1>& rhs) noexcept {
  return lhs.density_of_atmosphere_ == rhs.density_of_atmosphere_;
}

template <size_t ThermodynamicDim>
bool operator!=(const FixToAtmosphere<ThermodynamicDim>& lhs,
                const FixToAtmosphere<ThermodynamicDim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                   \
  template bool operator!=(                      \
      const FixToAtmosphere<GET_DIM(data)>& lhs, \
      const FixToAtmosphere<GET_DIM(data)>& rhs) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1))

#undef GET_DIM
#undef INSTANTIATION

}  // namespace VariableFixing
