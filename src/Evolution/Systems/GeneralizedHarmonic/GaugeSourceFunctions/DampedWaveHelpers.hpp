// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace gh {
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
template <typename DataType, size_t SpatialDim, typename Frame>
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
template <typename DataType, size_t SpatialDim, typename Frame>
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

}  // namespace DampedHarmonicGauge_detail
}  // namespace gauges
}  // namespace gh
