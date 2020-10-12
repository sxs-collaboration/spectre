// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RelativisticEuler/Valencia/TimeDerivativeTerms.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace RelativisticEuler::Valencia {
namespace detail {
template <size_t Dim>
void fluxes_impl(
    gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> tilde_d_flux,
    gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> tilde_tau_flux,
    gsl::not_null<tnsr::Ij<DataVector, Dim, Frame::Inertial>*> tilde_s_flux,
    gsl::not_null<DataVector*> transport_velocity_I,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& tilde_s,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& spatial_velocity,
    const DataVector& p_alpha_sqrt_det_g) noexcept;

template <size_t Dim>
void sources_impl(
    gsl::not_null<Scalar<DataVector>*> source_tilde_tau,
    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> source_tilde_s,
    gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> tilde_s_M,
    gsl::not_null<tnsr::II<DataVector, Dim, Frame::Inertial>*> tilde_s_MN,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& pressure, const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, Dim, Frame::Inertial>& d_shift,
    const tnsr::ijj<DataVector, Dim, Frame::Inertial>& d_spatial_metric,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>&
        extrinsic_curvature) noexcept;
}  // namespace detail

template <size_t Dim>
void TimeDerivativeTerms<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        non_flux_terms_dt_tilde_s,

    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        tilde_d_flux,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        tilde_tau_flux,
    const gsl::not_null<tnsr::Ij<DataVector, Dim, Frame::Inertial>*>
        tilde_s_flux,

    // For fluxes
    const gsl::not_null<Scalar<DataVector>*>
        pressure_lapse_sqrt_det_spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> transport_velocity,

    // For sources
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> tilde_s_up,
    const gsl::not_null<tnsr::II<DataVector, Dim, Frame::Inertial>*>
        densitized_stress,

    // For fluxes and sources
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& tilde_s,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const Scalar<DataVector>& pressure,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& spatial_velocity,

    // For sources
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, Dim, Frame::Inertial>& d_shift,
    const tnsr::ijj<DataVector, Dim, Frame::Inertial>& d_spatial_metric,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inv_spatial_metric,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>&
        extrinsic_curvature) noexcept {
  get(*pressure_lapse_sqrt_det_spatial_metric) =
      get(sqrt_det_spatial_metric) * get(lapse) * get(pressure);

  detail::fluxes_impl(tilde_d_flux, tilde_tau_flux, tilde_s_flux,
                      make_not_null(&get(*transport_velocity)), tilde_d,
                      tilde_tau, tilde_s, lapse, shift, spatial_velocity,
                      get(*pressure_lapse_sqrt_det_spatial_metric));

  get(*non_flux_terms_dt_tilde_d) = 0.0;
  detail::sources_impl(non_flux_terms_dt_tilde_tau, non_flux_terms_dt_tilde_s,
                       tilde_s_up, densitized_stress, tilde_d, tilde_tau,
                       tilde_s, spatial_velocity, pressure, lapse, d_lapse,
                       d_shift, d_spatial_metric, inv_spatial_metric,
                       sqrt_det_spatial_metric, extrinsic_curvature);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(r, data) template struct TimeDerivativeTerms<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace RelativisticEuler::Valencia
/// \endcond
