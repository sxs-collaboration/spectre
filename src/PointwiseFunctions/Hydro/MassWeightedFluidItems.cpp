// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "MassWeightedFluidItems.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace hydro {

template <typename DataType, size_t Dim, typename Frame>
void u_lower_t(const gsl::not_null<Scalar<DataType>*> result,
               const Scalar<DataType>& lorentz_factor,
               const tnsr::I<DataType, Dim, Frame>& spatial_velocity,
               const tnsr::ii<DataType, Dim, Frame>& spatial_metric,
               const Scalar<DataType>& lapse,
               const tnsr::I<DataType, Dim, Frame>& shift) {
  dot_product(result, spatial_velocity, shift, spatial_metric);
  result->get() = get(lorentz_factor) * (get(lapse) * (-1.0) + result->get());
}

template <typename DataType>
void mass_weighted_internal_energy(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& tilde_d,
    const Scalar<DataType>& specific_internal_energy) {
  result->get() = get(tilde_d) * get(specific_internal_energy);
}

template <typename DataType>
void mass_weighted_kinetic_energy(const gsl::not_null<Scalar<DataType>*> result,
                                  const Scalar<DataType>& tilde_d,
                                  const Scalar<DataType>& lorentz_factor) {
  result->get() = get(tilde_d) * (get(lorentz_factor) - 1.0);
}

template <typename DataType, size_t Dim, typename Fr>
void tilde_d_unbound_ut_criterion(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& tilde_d, const Scalar<DataType>& lorentz_factor,
    const tnsr::I<DataType, Dim, Fr>& spatial_velocity,
    const tnsr::ii<DataType, Dim, Fr>& spatial_metric,
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Fr>& shift) {
  u_lower_t(result, lorentz_factor, spatial_velocity, spatial_metric, lapse,
            shift);
  result->get() = get(tilde_d) * step_function(-1.0 - result->get());
}

template <domain::ObjectLabel Label, typename DataType, size_t Dim, typename Fr>
void mass_weighted_coords(
    const gsl::not_null<tnsr::I<DataType, Dim, Fr>*> result,
    const Scalar<DataType>& tilde_d,
    const tnsr::I<DataType, Dim, Frame::Grid>& grid_coords,
    const tnsr::I<DataType, Dim, Fr>& compute_coords) {
  for (size_t i = 0; i < Dim; i++) {
    result->get(i) = get(tilde_d) * (compute_coords.get(i));
    switch (Label) {
      case ::domain::ObjectLabel::A:
        result->get(i) *= step_function(get<0>(grid_coords));
        break;
      case ::domain::ObjectLabel::B:
        result->get(i) *= step_function(get<0>(grid_coords) * (-1.0));
        break;
      default:
        break;
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void u_lower_t(                                                     \
      const gsl::not_null<Scalar<DataVector>*> result,                         \
      const Scalar<DataVector>& lorentz_factor,                                \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& spatial_velocity, \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>& spatial_metric,  \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& shift);           \
  template void tilde_d_unbound_ut_criterion(                                  \
      const gsl::not_null<Scalar<DataVector>*> result,                         \
      const Scalar<DataVector>& tilde_d,                                       \
      const Scalar<DataVector>& lorentz_factor,                                \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& spatial_velocity, \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>& spatial_metric,  \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& shift);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define OBJECT(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template void mass_weighted_coords<OBJECT(data)>(                         \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*> \
          result,                                                           \
      const Scalar<DataVector>& tilde_d,                                    \
      const tnsr::I<DataVector, DIM(data), Frame::Grid>& dg_grid_coords,    \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& dg_coords);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3),
                        (::domain::ObjectLabel::None, ::domain::ObjectLabel::A,
                         ::domain::ObjectLabel::B))

#undef DIM
#undef OBJECT
#undef INSTANTIATE

template void mass_weighted_internal_energy(
    const gsl::not_null<Scalar<DataVector>*> result,
    const Scalar<DataVector>& tilde_d,
    const Scalar<DataVector>& specific_internal_energy);
template void mass_weighted_kinetic_energy(
    const gsl::not_null<Scalar<DataVector>*> result,
    const Scalar<DataVector>& tilde_d,
    const Scalar<DataVector>& lorentz_factor);

}  // namespace hydro
