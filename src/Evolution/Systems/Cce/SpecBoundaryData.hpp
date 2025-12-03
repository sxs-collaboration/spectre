// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/BoundaryDataTags.hpp"

/// \cond
class DataVector;
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace Cce {
/*
 * \brief Compute \f$\gamma_{i j}\f$, \f$\gamma^{i j}\f$,
 * \f$\partial_i \gamma_{j k}\f$, and
 * \f$\partial_t g_{i j}\f$ from input libsharp-compatible modal spatial
 * metric quantities.
 *
 * \details This function will apply a correction factor associated with a SpEC
 * bug.
 */
void cartesian_spatial_metric_and_derivatives_from_unnormalized_spec_modes(
    gsl::not_null<tnsr::ii<DataVector, 3>*> cartesian_spatial_metric,
    gsl::not_null<tnsr::II<DataVector, 3>*> inverse_cartesian_spatial_metric,
    gsl::not_null<tnsr::ijj<DataVector, 3>*> d_cartesian_spatial_metric,
    gsl::not_null<tnsr::ii<DataVector, 3>*> dt_cartesian_spatial_metric,
    gsl::not_null<Scalar<SpinWeighted<ComplexModalVector, 0>>*>
        interpolation_modal_buffer,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        interpolation_buffer,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> eth_buffer,
    gsl::not_null<Scalar<DataVector>*> radial_correction_factor,
    const tnsr::ii<ComplexModalVector, 3>& spatial_metric_coefficients,
    const tnsr::ii<ComplexModalVector, 3>& dr_spatial_metric_coefficients,
    const tnsr::ii<ComplexModalVector, 3>& dt_spatial_metric_coefficients,
    const CartesianiSphericalJ& inverse_cartesian_to_spherical_jacobian,
    const tnsr::I<DataVector, 3>& unit_cartesian_coords, size_t l_max);

/*!
 * \brief Compute \f$\beta^{i}\f$, \f$\partial_i \beta^{j}\f$, and
 * \f$\partial_t \beta^i\f$ from input libsharp-compatible modal spatial
 * metric quantities.
 *
 * \details This function will apply a correction factor associated with a SpEC
 * bug.
 */
void cartesian_shift_and_derivatives_from_unnormalized_spec_modes(
    gsl::not_null<tnsr::I<DataVector, 3>*> cartesian_shift,
    gsl::not_null<tnsr::iJ<DataVector, 3>*> d_cartesian_shift,
    gsl::not_null<tnsr::I<DataVector, 3>*> dt_cartesian_shift,
    gsl::not_null<Scalar<SpinWeighted<ComplexModalVector, 0>>*>
        interpolation_modal_buffer,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        interpolation_buffer,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> eth_buffer,
    const tnsr::I<ComplexModalVector, 3>& shift_coefficients,
    const tnsr::I<ComplexModalVector, 3>& dr_shift_coefficients,
    const tnsr::I<ComplexModalVector, 3>& dt_shift_coefficients,
    const CartesianiSphericalJ& inverse_cartesian_to_spherical_jacobian,
    const Scalar<DataVector>& radial_derivative_correction_factor,
    size_t l_max);

/*!
 * \brief Compute \f$\alpha\f$, \f$\partial_i \alpha\f$, and
 * \f$\partial_t \beta^i\f$ from input libsharp-compatible modal spatial
 * metric quantities.
 *
 * \details This function will apply a correction factor associated with a SpEC
 * bug.
 */
void cartesian_lapse_and_derivatives_from_unnormalized_spec_modes(
    gsl::not_null<Scalar<DataVector>*> cartesian_lapse,
    gsl::not_null<tnsr::i<DataVector, 3>*> d_cartesian_lapse,
    gsl::not_null<Scalar<DataVector>*> dt_cartesian_lapse,
    gsl::not_null<Scalar<SpinWeighted<ComplexModalVector, 0>>*>
        interpolation_modal_buffer,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        interpolation_buffer,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> eth_buffer,
    const Scalar<ComplexModalVector>& lapse_coefficients,
    const Scalar<ComplexModalVector>& dr_lapse_coefficients,
    const Scalar<ComplexModalVector>& dt_lapse_coefficients,
    const CartesianiSphericalJ& inverse_cartesian_to_spherical_jacobian,
    const Scalar<DataVector>& radial_derivative_correction_factor,
    size_t l_max);
}  // namespace Cce
