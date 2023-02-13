// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/ComovingMagneticField.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace hydro {

template <typename DataType>
void comoving_magnetic_field_one_form(
    const gsl::not_null<tnsr::a<DataType, 3>*> result,
    const tnsr::i<DataType, 3>& spatial_velocity_one_form,
    const tnsr::i<DataType, 3>& magnetic_field_one_form,
    const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataType>& lorentz_factor, const tnsr::I<DataType, 3>& shift,
    const Scalar<DataType>& lapse) {
  destructive_resize_components(result, get_size(get(lapse)));
  get<0>(*result) = -get(lapse) * get(lorentz_factor) *
                    get(magnetic_field_dot_spatial_velocity);
  for (size_t i = 0; i < 3; ++i) {
    result->get(i + 1) = magnetic_field_one_form.get(i) / get(lorentz_factor) +
                         get(lorentz_factor) *
                             get(magnetic_field_dot_spatial_velocity) *
                             spatial_velocity_one_form.get(i);
    get<0>(*result) += shift.get(i) * result->get(i + 1);
  }
}

template <typename DataType>
tnsr::a<DataType, 3> comoving_magnetic_field_one_form(
    const tnsr::i<DataType, 3>& spatial_velocity_one_form,
    const tnsr::i<DataType, 3>& magnetic_field_one_form,
    const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataType>& lorentz_factor, const tnsr::I<DataType, 3>& shift,
    const Scalar<DataType>& lapse) {
  tnsr::a<DataType, 3> result{};
  comoving_magnetic_field_one_form(
      make_not_null(&result), spatial_velocity_one_form,
      magnetic_field_one_form, magnetic_field_dot_spatial_velocity,
      lorentz_factor, shift, lapse);
  return result;
}

template <typename DataType>
void comoving_magnetic_field_squared(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& magnetic_field_squared,
    const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataType>& lorentz_factor) {
  get(*result) = get(magnetic_field_squared) / square(get(lorentz_factor)) +
                 square(get(magnetic_field_dot_spatial_velocity));
}

template <typename DataType>
Scalar<DataType> comoving_magnetic_field_squared(
    const Scalar<DataType>& magnetic_field_squared,
    const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataType>& lorentz_factor) {
  Scalar<DataType> result{};
  comoving_magnetic_field_squared(
      make_not_null(&result), magnetic_field_squared,
      magnetic_field_dot_spatial_velocity, lorentz_factor);
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data)                                               \
  template void comoving_magnetic_field_one_form(                            \
      const gsl::not_null<tnsr::a<DTYPE(data), 3>*>                          \
          comoving_magnetic_field_one_form_result,                           \
      const tnsr::i<DTYPE(data), 3>&, const tnsr::i<DTYPE(data), 3>&,        \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,                \
      const tnsr::I<DTYPE(data), 3>&, const Scalar<DTYPE(data)>&);           \
  template tnsr::a<DTYPE(data), 3> comoving_magnetic_field_one_form(         \
      const tnsr::i<DTYPE(data), 3>&, const tnsr::i<DTYPE(data), 3>&,        \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,                \
      const tnsr::I<DTYPE(data), 3>&, const Scalar<DTYPE(data)>&);           \
  template void comoving_magnetic_field_squared(                             \
      const gsl::not_null<Scalar<DTYPE(data)>*>, const Scalar<DTYPE(data)>&, \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&);               \
  template Scalar<DTYPE(data)> comoving_magnetic_field_squared(              \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,                \
      const Scalar<DTYPE(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATION, (double, DataVector))

#undef DTYPE
#undef INSTANTIATION

}  // namespace hydro
