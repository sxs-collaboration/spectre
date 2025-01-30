// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/StressEnergy.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/Hydro/ComovingMagneticField.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <typename DataType>
void four_velocity(const gsl::not_null<tnsr::A<DataType, 3>*> result,
                   const tnsr::I<DataType, 3>& spatial_velocity,
                   const tnsr::I<DataType, 3>& shift,
                   const Scalar<DataType>& lorentz_factor,
                   const Scalar<DataType>& lapse) {
  get<0>(*result) = get(lorentz_factor) / get(lapse);
  for (size_t i = 0; i < 3; ++i) {
    result->get(i + 1) =
        get<0>(*result) * (get(lapse) * spatial_velocity.get(i) - shift.get(i));
  }
}

template <typename DataType>
tnsr::A<DataType, 3> four_velocity(const tnsr::I<DataType, 3>& spatial_velocity,
                                   const tnsr::I<DataType, 3>& shift,
                                   const Scalar<DataType>& lorentz_factor,
                                   const Scalar<DataType>& lapse) {
  tnsr::A<DataType, 3> result{};
  four_velocity(make_not_null(&result), spatial_velocity, shift, lorentz_factor,
                lapse);
  return result;
}
}  // namespace

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

template <typename DataType>
void stress_energy_tensor(
    const gsl::not_null<tnsr::AA<DataType, 3>*> result,
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& pressure, const Scalar<DataType>& lorentz_factor,
    const Scalar<DataType>& lapse,
    const Scalar<DataType>& comoving_magnetic_field_magnitude,
    const tnsr::I<DataType, 3>& spatial_velocity,
    const tnsr::I<DataType, 3>& shift,
    const tnsr::I<DataType, 3>& magnetic_field,
    const tnsr::ii<DataType, 3>& spatial_metric,
    const tnsr::II<DataType, 3>& inverse_spatial_metric) {
  // Preallocate to minimize number of allocations.
  TempBuffer<tmpl::list<::Tags::TempScalar<0, DataType>,
                        ::Tags::TempScalar<1, DataType>,
                        ::Tags::TempScalar<2, DataType>,
                        ::Tags::TempA<3, 3, Frame::Inertial, DataType>,
                        ::Tags::TempA<4, 3, Frame::Inertial, DataType>,
                        ::Tags::TempAA<5, 3, Frame::Inertial, DataType>>>
      buffer(get_size(get(rest_mass_density)));

  auto& magnetic_field_dot_spatial_velocity =
      get<::Tags::TempScalar<0, DataType>>(buffer);
  auto& rho_h_star = get<::Tags::TempScalar<1, DataType>>(buffer);
  auto& p_star = get<::Tags::TempScalar<2, DataType>>(buffer);
  auto& comoving_magnetic_field_v =
      get<::Tags::TempA<3, 3, Frame::Inertial, DataType>>(buffer);
  auto& four_velocity_v =
      get<::Tags::TempA<4, 3, Frame::Inertial, DataType>>(buffer);
  auto& inverse_spacetime_metric_v =
      get<::Tags::TempAA<5, 3, Frame::Inertial, DataType>>(buffer);

  gr::inverse_spacetime_metric(make_not_null(&inverse_spacetime_metric_v),
                               lapse, shift, inverse_spatial_metric);

  dot_product(make_not_null(&magnetic_field_dot_spatial_velocity),
              magnetic_field, spatial_velocity, spatial_metric);

  comoving_magnetic_field(make_not_null(&comoving_magnetic_field_v),
                          spatial_velocity, magnetic_field,
                          magnetic_field_dot_spatial_velocity, lorentz_factor,
                          shift, lapse);

  four_velocity(make_not_null(&four_velocity_v), spatial_velocity, shift,
                lorentz_factor, lapse);

  get(rho_h_star) = (get(rest_mass_density) +
                     get(rest_mass_density) * get(specific_internal_energy)) +
                    get(pressure) +
                    square(get(comoving_magnetic_field_magnitude));

  get(p_star) =
      get(pressure) + 0.5 * square(get(comoving_magnetic_field_magnitude));

  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = i; j < 4; ++j) {
      result->get(i, j) =
          get(rho_h_star) * four_velocity_v.get(i) * four_velocity_v.get(j) +
          get(p_star) * inverse_spacetime_metric_v.get(i, j) -
          comoving_magnetic_field_v.get(i) * comoving_magnetic_field_v.get(j);
    }
  }
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data)                                              \
  template void energy_density(                                             \
      gsl::not_null<Scalar<DTYPE(data)>*>, const Scalar<DTYPE(data)>&,      \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,               \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,               \
      const Scalar<DTYPE(data)>&);                                          \
  template void momentum_density(                                           \
      gsl::not_null<tnsr::I<DTYPE(data), 3>*>, const Scalar<DTYPE(data)>&,  \
      const Scalar<DTYPE(data)>&, const tnsr::I<DTYPE(data), 3>&,           \
      const Scalar<DTYPE(data)>&, const tnsr::I<DTYPE(data), 3>&,           \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&);              \
  template void stress_trace(                                               \
      gsl::not_null<Scalar<DTYPE(data)>*>, const Scalar<DTYPE(data)>&,      \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,               \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,               \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&);              \
  template void stress_energy_tensor(                                       \
      gsl::not_null<tnsr::AA<DTYPE(data), 3>*>, const Scalar<DTYPE(data)>&, \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,               \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,               \
      const Scalar<DTYPE(data)>&, const tnsr::I<DTYPE(data), 3>&,           \
      const tnsr::I<DTYPE(data), 3>&, const tnsr::I<DTYPE(data), 3>&,       \
      const tnsr::ii<DTYPE(data), 3>&, const tnsr::II<DTYPE(data), 3>&);

GENERATE_INSTANTIATIONS(INSTANTIATION, (double, DataVector))

#undef DTYPE
#undef INSTANTIATION

}  // namespace hydro
