// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/Fluxes.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree {

namespace detail {
void fluxes_impl(
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_e_flux,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_psi_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_phi_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_q_flux,

    // Temporaries
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        lapse_times_electric_field_one_form,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        lapse_times_magnetic_field_one_form,

    // extra args
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_psi, const Scalar<DataVector>& tilde_phi,
    const Scalar<DataVector>& tilde_q,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_j,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric) {
  for (size_t j = 0; j < 3; ++j) {
    tilde_psi_flux->get(j) =
        -shift.get(j) * get(tilde_psi) + get(lapse) * tilde_e.get(j);

    tilde_phi_flux->get(j) =
        -shift.get(j) * get(tilde_phi) + get(lapse) * tilde_b.get(j);

    tilde_q_flux->get(j) = tilde_j.get(j) - shift.get(j) * get(tilde_q);

    for (size_t i = 0; i < 3; ++i) {
      tilde_e_flux->get(j, i) =
          -shift.get(j) * tilde_e.get(i) +
          get(lapse) * inv_spatial_metric.get(j, i) * get(tilde_psi);

      tilde_b_flux->get(j, i) =
          -shift.get(j) * tilde_b.get(i) +
          get(lapse) * inv_spatial_metric.get(j, i) * get(tilde_phi);
    }
  }

  for (LeviCivitaIterator<3> it; it; ++it) {
    const auto& i = it[0];
    const auto& j = it[1];
    const auto& k = it[2];

    tilde_e_flux->get(j, i) +=
        -it.sign() * lapse_times_magnetic_field_one_form.get(k);
    tilde_b_flux->get(j, i) +=
        it.sign() * lapse_times_electric_field_one_form.get(k);
  }
}
}  // namespace detail

void Fluxes::apply(
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_e_flux,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_psi_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_phi_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_q_flux,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_psi, const Scalar<DataVector>& tilde_phi,
    const Scalar<DataVector>& tilde_q,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_j,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric) {
  Variables<tmpl::list<::Tags::Tempi<0, 3>, ::Tags::Tempi<1, 3>,
                       ::Tags::TempScalar<0>>>
      buffer{get(lapse).size()};

  auto& lapse_times_electric_field_one_form = get<::Tags::Tempi<0, 3>>(buffer);
  auto& lapse_times_magnetic_field_one_form = get<::Tags::Tempi<1, 3>>(buffer);
  raise_or_lower_index(make_not_null(&lapse_times_electric_field_one_form),
                       tilde_e, spatial_metric);
  raise_or_lower_index(make_not_null(&lapse_times_magnetic_field_one_form),
                       tilde_b, spatial_metric);

  auto& lapse_over_sqrt_det_spatial_metric = get<::Tags::TempScalar<0>>(buffer);
  get(lapse_over_sqrt_det_spatial_metric) =
      get(lapse) / get(sqrt_det_spatial_metric);
  for (size_t i = 0; i < 3; ++i) {
    lapse_times_electric_field_one_form.get(i) *=
        get(lapse_over_sqrt_det_spatial_metric);
    lapse_times_magnetic_field_one_form.get(i) *=
        get(lapse_over_sqrt_det_spatial_metric);
  }

  detail::fluxes_impl(
      tilde_e_flux, tilde_b_flux, tilde_psi_flux, tilde_phi_flux, tilde_q_flux,
      // temporaries
      lapse_times_electric_field_one_form, lapse_times_magnetic_field_one_form,
      // extra args
      tilde_e, tilde_b, tilde_psi, tilde_phi, tilde_q, tilde_j, lapse, shift,
      inv_spatial_metric);
}

}  // namespace ForceFree
