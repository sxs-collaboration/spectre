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

template <typename DataType>
void rotational_shift(
    gsl::not_null<tnsr::I<DataType, 3>*> result,
    const tnsr::I<DataType, 3>& shift,
    const tnsr::I<DataType, 3>& spatial_rotational_killing_vector);
template <typename DataType>
tnsr::I<DataType, 3> rotational_shift(
    const tnsr::I<DataType, 3>& shift,
    const tnsr::I<DataType, 3>& spatial_rotational_killing_vector);
template <typename DataType>
void rotational_shift_stress(gsl::not_null<tnsr::Ij<DataType, 3>*> result,
                             const tnsr::I<DataType, 3>& rotational_shift,
                             const Scalar<DataType>& lapse,
                             const tnsr::ii<DataType, 3>& spatial_metric);
template <typename DataType>
tnsr::Ij<DataType, 3> rotational_shift_stress(
    const tnsr::I<DataType, 3>& rotational_shift, const Scalar<DataType>& lapse,
    const tnsr::ii<DataType, 3>& spatial_metric);
template <typename DataType>
void derivative_rotational_shift_over_lapse(
    gsl::not_null<tnsr::iJ<DataType, 3>*> result,
    const tnsr::I<DataType, 3>& rotational_shift,
    const tnsr::iJ<DataType, 3>& deriv_of_shift, const Scalar<DataType>& lapse,
    const tnsr::i<DataType, 3>& deriv_of_lapse,
    const tnsr::iJ<DataType, 3>& deriv_of_spatial_rotational_killing_vector);
template <typename DataType>
tnsr::iJ<DataType, 3> derivative_rotational_shift_over_lapse(
    const tnsr::I<DataType, 3>& rotational_shift,
    const tnsr::iJ<DataType, 3>& deriv_of_shift, const Scalar<DataType>& lapse,
    const tnsr::i<DataType, 3>& deriv_of_lapse,
    const tnsr::iJ<DataType, 3>& deriv_of_spatial_rotational_killing_vector);
template <typename DataType>
void divergence_rotational_shift_stress(
    gsl::not_null<tnsr::i<DataType, 3>*> result,
    const tnsr::I<DataType, 3>& rotational_shift,
    const tnsr::iJ<DataType, 3>& derivative_rotational_shift_over_lapse,
    const Scalar<DataType>& lapse, const tnsr::ii<DataType, 3>& spatial_metric);
template <typename DataType>
tnsr::i<DataType, 3> divergence_rotational_shift_stress(
    const tnsr::I<DataType, 3>& rotational_shift,
    const tnsr::iJ<DataType, 3>& derivative_rotational_shift_over_lapse,
    const Scalar<DataType>& lapse, const tnsr::ii<DataType, 3>& spatial_metric);
template <typename DataType>
void enthalpy_density_squared(
    gsl::not_null<Scalar<DataType>*> result,
    const tnsr::I<DataType, 3>& rotational_shift, const Scalar<DataType>& lapse,
    const tnsr::i<DataType, 3>& velocity_potential_gradient,
    const tnsr::II<DataType, 3>& inverse_spatial_metric,
    double euler_enthalpy_constant);
template <typename DataType>
Scalar<DataType> enthalpy_density_squared(
    const tnsr::I<DataType, 3>& rotational_shift, const Scalar<DataType>& lapse,
    const tnsr::i<DataType, 3>& velocity_potential_gradient,
    const tnsr::II<DataType, 3>& inverse_spatial_metric,
    double euler_enthalpy_constant);
template <typename DataType>
void spatial_rotational_killing_vector(
    gsl::not_null<tnsr::I<DataType, 3>*> result, const tnsr::I<DataType, 3>& x,
    const Scalar<DataType>& local_angular_velocity_around_z,
    const Scalar<DataType>& determinant_spatial_metric);
template <typename DataType>
tnsr::I<DataType, 3> spatial_rotational_killing_vector(
    const tnsr::I<DataType, 3>& x,
    const Scalar<DataType>& local_angular_velocity_around_z,
    const Scalar<DataType>& sqrt_det_spatial_metric);
template <typename DataType>
void derivative_spatial_rotational_killing_vector(
    gsl::not_null<tnsr::iJ<DataType, 3>*> result, const tnsr::I<DataType, 3>& x,
    const Scalar<DataType>& local_angular_velocity_around_z,
    const Scalar<DataType>& determinant_spatial_metric);
template <typename DataType>
tnsr::iJ<DataType, 3> derivative_spatial_rotational_killing_vector(
    const tnsr::I<DataType, 3>& x,
    const Scalar<DataType>& local_angular_velocity_around_z,
    const Scalar<DataType>& sqrt_det_spatial_metric);
}  // namespace hydro::initial_data
