// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RelativisticEuler/Valencia/Fluxes.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare Tensor

namespace RelativisticEuler::Valencia {
namespace detail {
template <size_t Dim>
void fluxes_impl(
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        tilde_d_flux,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        tilde_tau_flux,
    const gsl::not_null<tnsr::Ij<DataVector, Dim, Frame::Inertial>*>
        tilde_s_flux,
    const gsl::not_null<DataVector*> transport_velocity_I,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& tilde_s,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& spatial_velocity,
    const DataVector& p_alpha_sqrt_det_g) noexcept {
  for (size_t i = 0; i < Dim; ++i) {
    *transport_velocity_I = get(lapse) * spatial_velocity.get(i) - shift.get(i);
    tilde_d_flux->get(i) = get(tilde_d) * *transport_velocity_I;
    tilde_tau_flux->get(i) = get(tilde_tau) * *transport_velocity_I +
                             p_alpha_sqrt_det_g * spatial_velocity.get(i);
    for (size_t j = 0; j < Dim; ++j) {
      tilde_s_flux->get(i, j) = tilde_s.get(j) * *transport_velocity_I;
    }
    tilde_s_flux->get(i, i) += p_alpha_sqrt_det_g;
  }
}
}  // namespace detail

template <size_t Dim>
void ComputeFluxes<Dim>::apply(
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        tilde_d_flux,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        tilde_tau_flux,
    const gsl::not_null<tnsr::Ij<DataVector, Dim, Frame::Inertial>*>
        tilde_s_flux,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
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
  detail::fluxes_impl(tilde_d_flux, tilde_tau_flux, tilde_s_flux,
                      make_not_null(&transport_velocity_I), tilde_d, tilde_tau,
                      tilde_s, lapse, shift, spatial_velocity,
                      p_alpha_sqrt_det_g);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                                 \
  template class ComputeFluxes<DIM(data)>;                                     \
  template void detail::fluxes_impl(                                           \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>          \
          tilde_d_flux,                                                        \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>          \
          tilde_tau_flux,                                                      \
      gsl::not_null<tnsr::Ij<DataVector, DIM(data), Frame::Inertial>*>         \
          tilde_s_flux,                                                        \
      gsl::not_null<DataVector*> transport_velocity_I,                         \
      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,  \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& tilde_s,          \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& shift,            \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& spatial_velocity, \
      const DataVector& p_alpha_sqrt_det_g) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM

}  // namespace RelativisticEuler::Valencia
