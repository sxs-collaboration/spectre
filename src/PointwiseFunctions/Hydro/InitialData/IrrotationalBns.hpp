// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"

#include "Utilities/Gsl.hpp"

/// Have to document this
namespace hydro::initial_data {

void rotational_shift(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& spatial_rotational_killing_vector);

tnsr::I<DataVector, 3> rotational_shift(
    const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& spatial_rotational_killing_vector);

void rotational_shift_stress(gsl::not_null<tnsr::Ij<DataVector, 3>*> result,
                             const tnsr::I<DataVector, 3>& rotational_shift,
                             const Scalar<DataVector>& lapse,
                             const tnsr::ii<DataVector, 3>& spatial_metric);

tnsr::Ij<DataVector, 3> rotational_shift_stress(
    const tnsr::I<DataVector, 3>& rotational_shift,
    const Scalar<DataVector>& lapse,
    const tnsr::ii<DataVector, 3>& spatial_metric);

void derivative_rotational_shift_over_lapse(
    gsl::not_null<tnsr::iJ<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& rotational_shift,
    const tnsr::iJ<DataVector, 3>& deriv_of_shift,
    const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3>& deriv_of_lapse,
    const tnsr::iJ<DataVector, 3>& deriv_of_spatial_rotational_killing_vector);

tnsr::iJ<DataVector, 3> derivative_rotational_shift_over_lapse(
    const tnsr::I<DataVector, 3>& rotational_shift,
    const tnsr::iJ<DataVector, 3>& deriv_of_shift,
    const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3>& deriv_of_lapse,
    const tnsr::iJ<DataVector, 3>& deriv_of_spatial_rotational_killing_vector);

void divergence_rotational_shift_stress(
    gsl::not_null<tnsr::i<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& rotational_shift,
    const tnsr::iJ<DataVector, 3>& derivative_rotational_shift_over_lapse,
    const Scalar<DataVector>& lapse,
    const tnsr::ii<DataVector, 3>& spatial_metric);

tnsr::i<DataVector, 3> divergence_rotational_shift_stress(
    const tnsr::I<DataVector, 3>& rotational_shift,
    const tnsr::iJ<DataVector, 3>& derivative_rotational_shift_over_lapse,
    const Scalar<DataVector>& lapse,
    const tnsr::ii<DataVector, 3>& spatial_metric);

void enthalpy_density_squared(
    gsl::not_null<Scalar<DataVector>*> result,
    const tnsr::I<DataVector, 3>& rotational_shift,
    const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3>& velocity_potential_gradient,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
    double euler_enthalpy_constant);

Scalar<DataVector> enthalpy_density_squared(
    const tnsr::I<DataVector, 3>& rotational_shift,
    const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3>& velocity_potential_gradient,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
    double euler_enthalpy_constant);

void spatial_rotational_killing_vector(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& local_angular_velocity_around_z,
    const Scalar<DataVector>& determinant_spatial_metric);

tnsr::I<DataVector, 3> spatial_rotational_killing_vector(
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& local_angular_velocity_around_z,
    const Scalar<DataVector>& sqrt_det_spatial_metric);

void derivative_spatial_rotational_killing_vector(
    gsl::not_null<tnsr::iJ<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& local_angular_velocity_around_z,
    const Scalar<DataVector>& determinant_spatial_metric);

tnsr::iJ<DataVector, 3> derivative_spatial_rotational_killing_vector(
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& local_angular_velocity_around_z,
    const Scalar<DataVector>& sqrt_det_spatial_metric);
}  // namespace hydro::initial_data
