// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/StressEnergy.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::GhValenciaDivClean {

namespace {
// spatial components of the four-velocity are:
// u^i = W(v^i - \beta^i / \alpha)
// so the down-index spatial components are
// u_i = W v_i
//
// u_0 = - \alpha W + \beta^i u_i
void four_velocity_one_form(
    const gsl::not_null<tnsr::a<DataVector, 3>*> four_velocity_one_form_result,
    const tnsr::i<DataVector, 3, Frame::Inertial>& spatial_velocity_one_form,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const Scalar<DataVector>& lapse) noexcept {
  get<0>(*four_velocity_one_form_result) = -get(lapse) * get(lorentz_factor);
  for (size_t i = 0; i < 3; ++i) {
    four_velocity_one_form_result->get(i + 1) =
        get(lorentz_factor) * spatial_velocity_one_form.get(i);
    get<0>(*four_velocity_one_form_result) +=
        shift.get(i) * four_velocity_one_form_result->get(i + 1);
  }
}

// spatial components of the comoving magnetic field vector are:
// b^i = B^i / W + B^j v^k \gamma_{j k} u^i
//
// time component is:
// b^0 = W B^j v^k \gamma_{j k} / \alpha
//
// Using the spacetime metric, the corresponding down-index components are
// b_i = B_i / W + B^j v^k \gamma_{j k} W v_i
// and
// b_0 = - \alpha W v^i B_i + \beta^i b_i
//
// and the square of the vector is:
// b^2 = B^i B^j \gamma_{i j} / W^2 + (B^i v^j \gamma_{i j})^2
void comoving_magnetic_field_one_form(
    const gsl::not_null<tnsr::a<DataVector, 3>*>
        comoving_magnetic_field_one_form_result,
    const tnsr::i<DataVector, 3>& spatial_velocity_one_form,
    const tnsr::i<DataVector, 3, Frame::Inertial>& magnetic_field_one_form,
    const Scalar<DataVector>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const Scalar<DataVector>& lapse) noexcept {
  get<0>(*comoving_magnetic_field_one_form_result) =
      -get(lapse) * get(lorentz_factor) *
      get(magnetic_field_dot_spatial_velocity);
  for (size_t i = 0; i < 3; ++i) {
    comoving_magnetic_field_one_form_result->get(i + 1) =
        magnetic_field_one_form.get(i) / get(lorentz_factor) +
        get(lorentz_factor) * get(magnetic_field_dot_spatial_velocity) *
            spatial_velocity_one_form.get(i);
    get<0>(*comoving_magnetic_field_one_form_result) +=
        shift.get(i) * comoving_magnetic_field_one_form_result->get(i + 1);
  }
 }
}  // namespace

void add_stress_energy_term_to_dt_pi(
    const gsl::not_null<tnsr::aa<DataVector, 3>*> dt_pi,
    const tnsr::aa<DataVector, 3>& trace_reversed_stress_energy,
    const Scalar<DataVector>& lapse) noexcept {
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = a; b < 4; ++b) {
      dt_pi->get(a, b) -=
          16.0 * M_PI * get(lapse) * trace_reversed_stress_energy.get(a, b);
    }
  }
}

void trace_reversed_stress_energy(
    const gsl::not_null<tnsr::aa<DataVector, 3>*> stress_energy,
    const gsl::not_null<tnsr::a<DataVector, 3>*> four_velocity_one_form_buffer,
    const gsl::not_null<tnsr::a<DataVector, 3>*>
        comoving_magnetic_field_one_form_buffer,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::i<DataVector, 3, Frame::Inertial>& spatial_velocity_one_form,
    const tnsr::i<DataVector, 3, Frame::Inertial>& magnetic_field_one_form,
    const Scalar<DataVector>& magnetic_field_squared,
    const Scalar<DataVector>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& one_over_w_squared,
    const Scalar<DataVector>& pressure,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const Scalar<DataVector>& lapse) noexcept {
  four_velocity_one_form(four_velocity_one_form_buffer,
                         spatial_velocity_one_form, lorentz_factor, shift,
                         lapse);

  // 3,3 component will be the last component assigned, so we can use it as a
  // temporary buffer
  auto& modified_enthalpy_times_rest_mass = get<3, 3>(*stress_energy);
  modified_enthalpy_times_rest_mass =
      get(rest_mass_density) * get(specific_enthalpy) +
      square(get(magnetic_field_dot_spatial_velocity)) +
      get(magnetic_field_squared) * get(one_over_w_squared);

  comoving_magnetic_field_one_form(
      comoving_magnetic_field_one_form_buffer, spatial_velocity_one_form,
      magnetic_field_one_form, magnetic_field_dot_spatial_velocity,
      lorentz_factor, shift, lapse);
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = a; b < 4; ++b) {
      stress_energy->get(a, b) =
          modified_enthalpy_times_rest_mass *
              four_velocity_one_form_buffer->get(a) *
              four_velocity_one_form_buffer->get(b) +
          (0.5 * modified_enthalpy_times_rest_mass - get(pressure)) *
              spacetime_metric.get(a, b) -
          comoving_magnetic_field_one_form_buffer->get(a) *
              comoving_magnetic_field_one_form_buffer->get(b);
    }
  }
}
}  // namespace grmhd::GhValenciaDivClean
