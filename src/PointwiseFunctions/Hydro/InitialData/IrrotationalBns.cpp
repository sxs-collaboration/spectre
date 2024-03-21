// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/InitialData/IrrotationalBns.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace hydro::initial_data {
template <typename DataType>
void rotational_shift(
    const gsl::not_null<tnsr::I<DataType, 3>*> result,
    const tnsr::I<DataType, 3>& shift,
    const tnsr::I<DataType, 3>& spatial_rotational_killing_vector) {
  ::tenex::evaluate<ti::I>(
      result, shift(ti::I) + spatial_rotational_killing_vector(ti::I));
}

template <typename DataType>
tnsr::I<DataType, 3> rotational_shift(
    const tnsr::I<DataType, 3>& shift,
    const tnsr::I<DataType, 3>& spatial_rotational_killing_vector) {
  tnsr::I<DataType, 3> buffer{};
  rotational_shift(make_not_null(&buffer), shift,
                   spatial_rotational_killing_vector);
  return buffer;
}

template <typename DataType>
void rotational_shift_stress(const gsl::not_null<tnsr::II<DataType, 3>*> result,
                             const tnsr::I<DataType, 3>& rotational_shift,
                             const Scalar<DataType>& lapse) {
  ::tenex::evaluate<ti::I, ti::J>(
      result,
      rotational_shift(ti::I) * rotational_shift(ti::J) / square(lapse()));
}

template <typename DataType>
tnsr::II<DataType, 3> rotational_shift_stress(
    const tnsr::I<DataType, 3>& rotational_shift,
    const Scalar<DataType>& lapse) {
  tnsr::II<DataType, 3> buffer{};
  rotational_shift_stress(make_not_null(&buffer), rotational_shift, lapse);
  return buffer;
}

template <typename DataType>
void derivative_rotational_shift_over_lapse(
    const gsl::not_null<tnsr::iJ<DataType, 3>*> result,
    const tnsr::I<DataType, 3>& rotational_shift,
    const tnsr::iJ<DataType, 3>& deriv_of_shift, const Scalar<DataType>& lapse,
    const tnsr::i<DataType, 3>& deriv_of_lapse,
    const tnsr::iJ<DataType, 3>& deriv_of_spatial_rotational_killing_vector) {
  ::tenex::evaluate<ti::i, ti::J>(
      result,
      (deriv_of_shift(ti::i, ti::J) +
       deriv_of_spatial_rotational_killing_vector(ti::i, ti::J)) /
              lapse() -
          rotational_shift(ti::J) * deriv_of_lapse(ti::i) / square(lapse()));
}

template <typename DataType>
tnsr::iJ<DataType, 3> derivative_rotational_shift_over_lapse(
    const tnsr::I<DataType, 3>& rotational_shift,
    const tnsr::iJ<DataType, 3>& deriv_of_shift, const Scalar<DataType>& lapse,
    const tnsr::i<DataType, 3>& deriv_of_lapse,
    const tnsr::iJ<DataType, 3>& deriv_of_spatial_rotational_killing_vector) {
  tnsr::iJ<DataType, 3> buffer{};
  derivative_rotational_shift_over_lapse(
      make_not_null(&buffer), rotational_shift, deriv_of_shift, lapse,
      deriv_of_lapse, deriv_of_spatial_rotational_killing_vector);
  return buffer;
}

template <typename DataType>
void divergence_rotational_shift_stress(
    const gsl::not_null<tnsr::i<DataType, 3>*> result,
    const tnsr::I<DataType, 3>& rotational_shift,
    const tnsr::iJ<DataType, 3>& derivative_rotational_shift_over_lapse,
    const Scalar<DataType>& lapse,
    const tnsr::ii<DataType, 3>& spatial_metric) {
  // This has symmetry properties that should be used
  ::tenex::evaluate<ti::k>(
      result, rotational_shift(ti::J) *
                      derivative_rotational_shift_over_lapse(ti::j, ti::I) *
                      spatial_metric(ti::i, ti::k) / lapse() +
                  rotational_shift(ti::I) * spatial_metric(ti::i, ti::k) *
                      derivative_rotational_shift_over_lapse(ti::j, ti::J) /
                      lapse());
}

template <typename DataType>
tnsr::i<DataType, 3> divergence_rotational_shift_stress(
    const tnsr::I<DataType, 3>& rotational_shift,
    const tnsr::iJ<DataType, 3>& derivative_rotational_shift_over_lapse,
    const Scalar<DataType>& lapse,
    const tnsr::ii<DataType, 3>& spatial_metric) {
  tnsr::i<DataType, 3> buffer{};
  divergence_rotational_shift_stress(make_not_null(&buffer), rotational_shift,
                                     derivative_rotational_shift_over_lapse,
                                     lapse, spatial_metric);
  return buffer;
}

template <typename DataType>
void enthalpy_density_squared(
    const gsl::not_null<Scalar<DataType>*> result,
    const tnsr::I<DataType, 3>& rotational_shift, const Scalar<DataType>& lapse,
    const tnsr::i<DataType, 3>& velocity_potential_gradient,
    const tnsr::II<DataType, 3>& inverse_spatial_metric,
    const double euler_enthalpy_constant) {
  return tenex::evaluate<>(
      result,
      square((euler_enthalpy_constant +
              rotational_shift(ti::I) * velocity_potential_gradient(ti::i)) /
             lapse()) -
          velocity_potential_gradient(ti::i) *
              velocity_potential_gradient(ti::j) *
              inverse_spatial_metric(ti::I, ti::J));
}

template <typename DataType>
Scalar<DataType> enthalpy_density_squared(
    const tnsr::I<DataType, 3>& rotational_shift, const Scalar<DataType>& lapse,
    const tnsr::i<DataType, 3>& velocity_potential_gradient,
    const tnsr::II<DataType, 3>& inverse_spatial_metric,
    const double euler_enthalpy_constant) {
  Scalar<DataType> buffer{};
  enthalpy_density_squared(make_not_null(&buffer), rotational_shift, lapse,
                           velocity_potential_gradient, inverse_spatial_metric,
                           euler_enthalpy_constant);
  return buffer;
}

template <typename DataType>
void spatial_rotational_killing_vector(
    const gsl::not_null<tnsr::I<DataType, 3>*> result,
    const tnsr::I<DataType, 3>& x,
    const Scalar<DataType>& local_angular_velocity_around_z,
    const Scalar<DataType>& sqrt_det_spatial_metric) {
  // Cross product involves volume element in arbitrary coordinates
  result->get(0) = -get(sqrt_det_spatial_metric) * get<1>(x) *
                   get(local_angular_velocity_around_z);
  result->get(1) = get(sqrt_det_spatial_metric) * get<0>(x) *
                   get(local_angular_velocity_around_z);
  result->get(2) = make_with_value<DataType>(sqrt_det_spatial_metric, 0.0);
}

template <typename DataType>
tnsr::I<DataType, 3> spatial_rotational_killing_vector(
    const tnsr::I<DataType, 3>& x,
    const Scalar<DataType>& local_angular_velocity_around_z,
    const Scalar<DataType>& sqrt_det_spatial_metric) {
  tnsr::I<DataType, 3> buffer{};
  spatial_rotational_killing_vector(make_not_null(&buffer), x,
                                    local_angular_velocity_around_z,
                                    sqrt_det_spatial_metric);
  return buffer;
}

template <typename DataType>
void derivative_spatial_rotational_killing_vector(
    const gsl::not_null<tnsr::iJ<DataType, 3>*> result,
    const tnsr::I<DataType, 3>& x,
    const Scalar<DataType>& /*local_angular_velocity_around_z*/,
    const Scalar<DataType>& /*sqrt_det_spatial_metric*/) {
  *result = make_with_value<tnsr::iJ<DataType, 3>>(get<0>(x), 0.0);
}

template <typename DataType>
tnsr::iJ<DataType, 3> derivative_spatial_rotational_killing_vector(
    const tnsr::I<DataType, 3>& x,
    const Scalar<DataType>& local_angular_velocity_around_z,
    const Scalar<DataType>& sqrt_det_spatial_metric) {
  tnsr::iJ<DataType, 3> buffer{};
  derivative_spatial_rotational_killing_vector(make_not_null(&buffer), x,
                                               local_angular_velocity_around_z,
                                               sqrt_det_spatial_metric);
  return buffer;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data)                                                \
  template void rotational_shift(                                             \
      gsl::not_null<tnsr::I<DTYPE(data), 3>*> result,                         \
      const tnsr::I<DTYPE(data), 3>& shift,                                   \
      const tnsr::I<DTYPE(data), 3>& spatial_rotational_killing_vector);      \
  template tnsr::I<DTYPE(data), 3> rotational_shift(                          \
      const tnsr::I<DTYPE(data), 3>& shift,                                   \
      const tnsr::I<DTYPE(data), 3>& spatial_rotational_killing_vector);      \
  template void rotational_shift_stress(                                      \
      gsl::not_null<tnsr::II<DTYPE(data), 3>*> result,                        \
      const tnsr::I<DTYPE(data), 3>& rotational_shift,                        \
      const Scalar<DTYPE(data)>& lapse);                                      \
  template tnsr::II<DTYPE(data), 3> rotational_shift_stress(                  \
      const tnsr::I<DTYPE(data), 3>& rotational_shift,                        \
      const Scalar<DTYPE(data)>& lapse);                                      \
  template void derivative_rotational_shift_over_lapse(                       \
      gsl::not_null<tnsr::iJ<DTYPE(data), 3>*> result,                        \
      const tnsr::I<DTYPE(data), 3>& rotational_shift,                        \
      const tnsr::iJ<DTYPE(data), 3>& deriv_of_shift,                         \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::i<DTYPE(data), 3>& deriv_of_lapse,                          \
      const tnsr::iJ<DTYPE(data), 3>&                                         \
          deriv_of_spatial_rotational_killing_vector);                        \
  template tnsr::iJ<DTYPE(data), 3> derivative_rotational_shift_over_lapse(   \
      const tnsr::I<DTYPE(data), 3>& rotational_shift,                        \
      const tnsr::iJ<DTYPE(data), 3>& deriv_of_shift,                         \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::i<DTYPE(data), 3>& deriv_of_lapse,                          \
      const tnsr::iJ<DTYPE(data), 3>&                                         \
          deriv_of_spatial_rotational_killing_vector);                        \
  template void divergence_rotational_shift_stress(                           \
      gsl::not_null<tnsr::i<DTYPE(data), 3>*> result,                         \
      const tnsr::I<DTYPE(data), 3>& rotational_shift,                        \
      const tnsr::iJ<DTYPE(data), 3>& derivative_rotational_shift_over_lapse, \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::ii<DTYPE(data), 3>& spatial_metric);                        \
  template tnsr::i<DTYPE(data), 3> divergence_rotational_shift_stress(        \
      const tnsr::I<DTYPE(data), 3>& rotational_shift,                        \
      const tnsr::iJ<DTYPE(data), 3>& derivative_rotational_shift_over_lapse, \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::ii<DTYPE(data), 3>& spatial_metric);                        \
  template void enthalpy_density_squared(                                     \
      gsl::not_null<Scalar<DTYPE(data)>*> result,                             \
      const tnsr::I<DTYPE(data), 3>& rotational_shift,                        \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::i<DTYPE(data), 3>& velocity_potential_gradient,             \
      const tnsr::II<DTYPE(data), 3>& inverse_spatial_metric,                 \
      double euler_enthalpy_constant);                                        \
  template Scalar<DTYPE(data)> enthalpy_density_squared(                      \
      const tnsr::I<DTYPE(data), 3>& rotational_shift,                        \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::i<DTYPE(data), 3>& velocity_potential_gradient,             \
      const tnsr::II<DTYPE(data), 3>& inverse_spatial_metric,                 \
      double euler_enthalpy_constant);                                        \
  template void spatial_rotational_killing_vector(                            \
      gsl::not_null<tnsr::I<DTYPE(data), 3>*> result,                         \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      const Scalar<DTYPE(data)>& local_angular_velocity_around_z,             \
      const Scalar<DTYPE(data)>& determinant_spatial_metric);                 \
  template tnsr::I<DTYPE(data), 3> spatial_rotational_killing_vector(         \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      const Scalar<DTYPE(data)>& local_angular_velocity_around_z,             \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric);                    \
  template void derivative_spatial_rotational_killing_vector(                 \
      gsl::not_null<tnsr::iJ<DTYPE(data), 3>*> result,                        \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      const Scalar<DTYPE(data)>& local_angular_velocity_around_z,             \
      const Scalar<DTYPE(data)>& determinant_spatial_metric);                 \
  template tnsr::iJ<DTYPE(data), 3>                                           \
  derivative_spatial_rotational_killing_vector(                               \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      const Scalar<DTYPE(data)>& local_angular_velocity_around_z,             \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric);

GENERATE_INSTANTIATIONS(INSTANTIATION, (double, DataVector))

#undef DTYPE
#undef INSTANTIATION

}  // namespace hydro::initial_data
