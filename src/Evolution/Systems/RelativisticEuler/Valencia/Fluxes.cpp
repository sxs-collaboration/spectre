// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RelativisticEuler/Valencia/Fluxes.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace RelativisticEuler {
namespace Valencia {

template <size_t Dim>
void fluxes(const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
                tilde_d_flux,
            const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
                tilde_tau_flux,
            const gsl::not_null<tnsr::Ij<DataVector, Dim, Frame::Inertial>*>
                tilde_s_flux,
            const Scalar<DataVector>& tilde_d,
            const Scalar<DataVector>& tilde_tau,
            const tnsr::i<DataVector, Dim, Frame::Inertial>& tilde_s,
            const Scalar<DataVector>& lapse,
            const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
            const Scalar<DataVector>& sqrt_det_spatial_metric,
            const Scalar<DataVector>& pressure,
            const tnsr::I<DataVector, Dim, Frame::Inertial>&
                spatial_velocity) noexcept {
  const DataVector p_alpha_sqrt_det_g =
      get(sqrt_det_spatial_metric) * get(lapse) * get(pressure);
  // Outside the loop to save allocations
  DataVector transport_velocity_I(p_alpha_sqrt_det_g.size());
  for (size_t i = 0; i < Dim; ++i) {
    transport_velocity_I = get(lapse) * spatial_velocity.get(i) - shift.get(i);
    tilde_d_flux->get(i) = get(tilde_d) * transport_velocity_I;
    tilde_tau_flux->get(i) = get(tilde_tau) * transport_velocity_I +
                             p_alpha_sqrt_det_g * spatial_velocity.get(i);
    for (size_t j = 0; j < Dim; ++j) {
      tilde_s_flux->get(i, j) = tilde_s.get(j) * transport_velocity_I;
    }
    tilde_s_flux->get(i, i) += p_alpha_sqrt_det_g;
  }
}
}  // namespace Valencia
}  // namespace RelativisticEuler

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                                \
  template void RelativisticEuler::Valencia::fluxes(                          \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>   \
          tilde_d_flux,                                                       \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>   \
          tilde_tau_flux,                                                     \
      const gsl::not_null<tnsr::Ij<DataVector, DIM(data), Frame::Inertial>*>  \
          tilde_s_flux,                                                       \
      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau, \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& tilde_s,         \
      const Scalar<DataVector>& lapse,                                        \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& shift,           \
      const Scalar<DataVector>& sqrt_det_spatial_metric,                      \
      const Scalar<DataVector>& pressure,                                     \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&                  \
          spatial_velocity) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
/// \endcond
