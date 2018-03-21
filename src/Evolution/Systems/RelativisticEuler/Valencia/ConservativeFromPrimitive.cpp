// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RelativisticEuler/Valencia/ConservativeFromPrimitive.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace RelativisticEuler {
namespace Valencia {

template <typename DataType, size_t Dim>
void conservative_from_primitive(
    const gsl::not_null<Scalar<DataType>*> tilde_d,
    const gsl::not_null<Scalar<DataType>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataType, Dim, Frame::Inertial>*> tilde_s,
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const tnsr::i<DataType, Dim, Frame::Inertial>& spatial_velocity_oneform,
    const Scalar<DataType>& spatial_velocity_squared,
    const Scalar<DataType>& lorentz_factor,
    const Scalar<DataType>& specific_enthalpy, const Scalar<DataType>& pressure,
    const Scalar<DataType>& sqrt_det_spatial_metric) noexcept {
  get(*tilde_d) = get(sqrt_det_spatial_metric) * get(rest_mass_density) *
                  get(lorentz_factor);

  get(*tilde_tau) = get(sqrt_det_spatial_metric) * square(get(lorentz_factor)) *
                    (get(rest_mass_density) *
                         (get(specific_internal_energy) +
                          get(spatial_velocity_squared) * get(lorentz_factor) /
                              (get(lorentz_factor) + 1.)) +
                     get(pressure) * get(spatial_velocity_squared));

  for (size_t i = 0; i < Dim; ++i) {
    tilde_s->get(i) = get(*tilde_d) * get(lorentz_factor) *
                      get(specific_enthalpy) * spatial_velocity_oneform.get(i);
  }
}

}  // namespace Valencia
}  // namespace RelativisticEuler

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data)                                               \
  template void RelativisticEuler::Valencia::conservative_from_primitive(    \
      const gsl::not_null<Scalar<DTYPE(data)>*> tilde_d,                     \
      const gsl::not_null<Scalar<DTYPE(data)>*> tilde_tau,                   \
      const gsl::not_null<tnsr::i<DTYPE(data), DIM(data), Frame::Inertial>*> \
          tilde_s,                                                           \
      const Scalar<DTYPE(data)>& rest_mass_density,                          \
      const Scalar<DTYPE(data)>& specific_internal_energy,                   \
      const tnsr::i<DTYPE(data), DIM(data), Frame::Inertial>&                \
          spatial_velocity_oneform,                                          \
      const Scalar<DTYPE(data)>& spatial_velocity_squared,                   \
      const Scalar<DTYPE(data)>& lorentz_factor,                             \
      const Scalar<DTYPE(data)>& specific_enthalpy,                          \
      const Scalar<DTYPE(data)>& pressure,                                   \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (double, DataVector), (1, 2, 3))

#undef INSTANTIATION
#undef DIM
#undef DTYPE
/// \endcond
