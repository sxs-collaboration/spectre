// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace GeneralizedHarmonic {
namespace gauges {
namespace DampedHarmonicGauge_detail {
/*
 * Spatial weight function used in the damped harmonic gauge source
 * function.
 *
 * The spatial weight function is:
 * \f{align*}{
 *   W(x^i) = \exp(- (r / \sigma_r)^2),
 * \f}
 *
 * where \f$r=\sqrt{x^i\delta_{ij}x^j}\f$ is the coordinate radius, and
 * \f$\sigma_r\f$ is the width of the Gaussian.
 *
 * This function can be written with an extra factor inside the exponent in
 * literature, e.g. \cite Deppe2018uye. We absorb that in \f$\sigma_r\f$.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void spatial_weight_function(gsl::not_null<Scalar<DataType>*> weight,
                             const tnsr::I<DataType, SpatialDim, Frame>& coords,
                             double sigma_r);

/*
 * Spacetime derivatives of the spatial weight function that enters the
 * damped harmonic gauge source function.
 *
 * Compute the derivatives:
 * \f{align*}{
 * \partial_a W(x^i)= \partial_a \exp(- (r/\sigma_r)^2)
 *                  = (-2 * x^i / \sigma_r^2) * exp(-(r/\sigma_r)^2)
 * \f}
 *
 * where \f$r=\sqrt{x^i\delta_{ij}x^j}\f$ is the coordinate radius, and
 * \f$\sigma_r\f$ is the width of the Gaussian. Since the weight function is
 * spatial, the time derivative is always zero.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_deriv_of_spatial_weight_function(
    gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> d4_weight,
    const tnsr::I<DataType, SpatialDim, Frame>& coords, double sigma_r,
    const Scalar<DataType>& weight_function);

/*
 * The log factor that appears in damped harmonic gauge source function.
 *
 * Calculates:  \f$ logF = \mathrm{log}(g^p/N) \f$.
 */
template <typename DataType>
void log_factor_metric_lapse(gsl::not_null<Scalar<DataType>*> logfac,
                             const Scalar<DataType>& lapse,
                             const Scalar<DataType>& sqrt_det_spatial_metric,
                             double exponent);

template <typename DataType>
Scalar<DataType> log_factor_metric_lapse(
    const Scalar<DataType>& lapse,
    const Scalar<DataType>& sqrt_det_spatial_metric, double exponent);

/*
 * Spacetime derivatives of the log factor that appears in the
 * damped harmonic gauge source function.
 *
 * Computes the spacetime derivatives:
 * \f{align*}{
 *  \partial_a logF = (p/g)\partial_a g - (1/N)\partial_a N
 * \f}
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_deriv_of_log_factor_metric_lapse(
    gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> d4_logfac,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi, double exponent);

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> spacetime_deriv_of_log_factor_metric_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi, double exponent);

/*
 * Spacetime derivatives of the log factor (that appears in
 * damped harmonic gauge source function), raised to an exponent.
 *
 * Computes the spacetime derivatives:
 * \f{align*}{
 *  \partial_a (logF)^q = q (logF)^{q-1} \partial_a (logF)
 * \f}
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_deriv_of_power_log_factor_metric_lapse(
    gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> d4_powlogfac,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi, double g_exponent,
    int exponent);

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame>
spacetime_deriv_of_power_log_factor_metric_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi, double g_exponent,
    int exponent);

}  // namespace DampedHarmonicGauge_detail
}  // namespace gauges
}  // namespace GeneralizedHarmonic
