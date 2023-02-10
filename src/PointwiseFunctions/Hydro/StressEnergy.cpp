// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/StressEnergy.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace hydro {

template <typename DataType>
void energy_density(gsl::not_null<Scalar<DataType>*> result,
                    const Scalar<DataType>& rest_mass_density,
                    const Scalar<DataType>& specific_enthalpy,
                    const Scalar<DataType>& pressure,
                    const Scalar<DataType>& lorentz_factor,
                    const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
                    const Scalar<DataType>& comoving_magnetic_field_squared) {
  *result = rest_mass_density;
  get(*result) *= get(specific_enthalpy);
  get(*result) += get(comoving_magnetic_field_squared);
  get(*result) -= square(get(magnetic_field_dot_spatial_velocity));
  get(*result) *= square(get(lorentz_factor));
  get(*result) -= get(pressure);
  get(*result) -= 0.5 * get(comoving_magnetic_field_squared);
}

template <typename DataType>
void momentum_density(
    gsl::not_null<tnsr::I<DataType, 3>*> result,
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_enthalpy,
    const tnsr::I<DataType, 3>& spatial_velocity,
    const Scalar<DataType>& lorentz_factor,
    const tnsr::I<DataType, 3>& magnetic_field,
    const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataType>& comoving_magnetic_field_squared) {
  get<0>(*result) = (get(rest_mass_density) * get(specific_enthalpy) +
                     get(comoving_magnetic_field_squared) -
                     square(get(magnetic_field_dot_spatial_velocity))) *
                    square(get(lorentz_factor));
  get<1>(*result) = get<0>(*result);
  get<2>(*result) = get<0>(*result);
  for (size_t d = 0; d < 3; ++d) {
    result->get(d) *= spatial_velocity.get(d);
    result->get(d) -=
        get(magnetic_field_dot_spatial_velocity) * magnetic_field.get(d);
  }
}

template <typename DataType>
void stress_trace(gsl::not_null<Scalar<DataType>*> result,
                  const Scalar<DataType>& rest_mass_density,
                  const Scalar<DataType>& specific_enthalpy,
                  const Scalar<DataType>& pressure,
                  const Scalar<DataType>& spatial_velocity_squared,
                  const Scalar<DataType>& lorentz_factor,
                  const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
                  const Scalar<DataType>& comoving_magnetic_field_squared) {
  get(*result) =
      3. * get(pressure) +
      get(rest_mass_density) * get(specific_enthalpy) *
          (square(get(lorentz_factor)) - 1.) +
      get(comoving_magnetic_field_squared) *
          (square(get(lorentz_factor)) * get(spatial_velocity_squared) + 0.5) -
      square(get(magnetic_field_dot_spatial_velocity)) *
          (square(get(lorentz_factor)) * get(spatial_velocity_squared) + 1.);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data)                                             \
  template void energy_density(                                            \
      gsl::not_null<Scalar<DTYPE(data)>*>, const Scalar<DTYPE(data)>&,     \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,              \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,              \
      const Scalar<DTYPE(data)>&);                                         \
  template void momentum_density(                                          \
      gsl::not_null<tnsr::I<DTYPE(data), 3>*>, const Scalar<DTYPE(data)>&, \
      const Scalar<DTYPE(data)>&, const tnsr::I<DTYPE(data), 3>&,          \
      const Scalar<DTYPE(data)>&, const tnsr::I<DTYPE(data), 3>&,          \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&);             \
  template void stress_trace(                                              \
      gsl::not_null<Scalar<DTYPE(data)>*>, const Scalar<DTYPE(data)>&,     \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,              \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,              \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATION, (double, DataVector))

#undef DTYPE
#undef INSTANTIATION

}  // namespace hydro
