// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Utilities/Gsl.hpp"

/// @{
/*!
 * \brief Compute the Euclidean area element from the inverse jacobian.
 *
 * \details The Euclidean area element of a surface \f$\Sigma\f$ with
 * constant logical coordinate \f$\xi^i\f$ is given by:
 *
 * \f{equation}
 * J^\Sigma = J \sqrt{\delta^{jk} (J^{-1})^i_j (J^{-1})^i_k}
 * \f}
 *
 * where \f$J^i_j = \partial x^i / \xi^j\f$ is the volume Jacobian with
 * determinant \f$J\f$ and inverse \f$(J^{-1})^i_j = \partial \xi^i / \partial
 * x^j\f$. The determinant of the inverse Jacobian can be passed as an argument
 * as an overload, otherwise it will be calculated.
 *
 * \note Time dependent maps are not implemented yet but can be added on
 * demand.
 *
 * \param inverse_jacobian_face The inverse Jacobian from the ElementLogical
 * frame to the Target frame sliced to the surface.
 * \param inverse_jacobian_determinant_face The determinant of the inverse
 * Jacobian sliced onto the surface. If not availabe, there exists an overload
 * that will calculate it from the inverse Jacobian.
 * \param direction The direction of the surface in the element.
 * */
template <size_t VolumeDim, typename TargetFrame>
void euclidean_area_element(
    const gsl::not_null<Scalar<DataVector>*> result,
    const InverseJacobian<DataVector, VolumeDim, Frame::ElementLogical,
                          TargetFrame>& inverse_jacobian_face,
    const Scalar<DataVector>& inverse_jacobian_determinant_face,
    const Direction<VolumeDim>& direction);

template <size_t VolumeDim, typename TargetFrame>
void euclidean_area_element(
    const gsl::not_null<Scalar<DataVector>*> result,
    const InverseJacobian<DataVector, VolumeDim, Frame::ElementLogical,
                          TargetFrame>& inverse_jacobian_face,
    const Direction<VolumeDim>& direction);

template <size_t VolumeDim, typename TargetFrame>
Scalar<DataVector> euclidean_area_element(
    const InverseJacobian<DataVector, VolumeDim, Frame::ElementLogical,
                          TargetFrame>& inverse_jacobian_face,
    const Scalar<DataVector>& inverse_jacobian_determinant_face,
    const Direction<VolumeDim>& direction);

template <size_t VolumeDim, typename TargetFrame>
Scalar<DataVector> euclidean_area_element(
    const InverseJacobian<DataVector, VolumeDim, Frame::ElementLogical,
                          TargetFrame>& inverse_jacobian_face,
    const Direction<VolumeDim>& direction);
/// @}

/// @{
/*!
 * \brief Compute the curved area element from the inverse jacobian.
 *
 * \details The area element of a surface \f$\Sigma\f$ with
 * constant logical coordinate \f$\xi^i\f$ is given by:
 *
 * \f{equation}
 * J^\Sigma = J \sqrt{\gamma} \sqrt{\gamma^{jk} (J^{-1})^i_j (J^{-1})^i_k}
 * \f}
 *
 * where \f$\sqrt{\gamma}\f$ is the determinant of the spatial metric and
 * \f$J^i_j = \partial x^i / \xi^j\f$ is the volume Jacobian with determinant
 * \f$J\f$ and inverse \f$(J^{-1})^i_j = \partial \xi^i / \partial x^j\f$. Note
 * that the square root in the expression above is the magnitude of the
 * unnormalized face normal, where \f$\gamma^{jk}\f$ is the inverse spatial
 * metric. The determinant of the inverse Jacobian can be passed as an argument
 * as an overload, otherwise it will be calculated.
 *
 * \param inverse_jacobian_face The inverse Jacobian from the ElementLogical
 * frame to the Target frame sliced to the surface.
 * \param inverse_jacobian_determinant_face The determinant of the inverse
 * Jacobian sliced onto the surface. If not availabe, there exists an overload
 * that will calculate it from the inverse Jacobian.
 * \param direction The direction of the surface in the element.
 * \param inverse_spatial_metric The inverse spatial metric sliced to the
 * surface.
 * \param sqrt_det_spatial_metric The square root of the determinant of the
 * spatial metric, see \ref gr::Tags::SqrtDetSpatialMetric.
 */
template <size_t VolumeDim, typename TargetFrame>
void area_element(
    const gsl::not_null<Scalar<DataVector>*> result,
    const InverseJacobian<DataVector, VolumeDim, Frame::ElementLogical,
                          TargetFrame>& inverse_jacobian_face,
    const Scalar<DataVector>& inverse_jacobian_determinant_face,
    const Direction<VolumeDim>& direction,
    const tnsr::II<DataVector, VolumeDim, TargetFrame>& inverse_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric);

template <size_t VolumeDim, typename TargetFrame>
void area_element(
    const gsl::not_null<Scalar<DataVector>*> result,
    const InverseJacobian<DataVector, VolumeDim, Frame::ElementLogical,
                          TargetFrame>& inverse_jacobian_face,
    const Direction<VolumeDim>& direction,
    const tnsr::II<DataVector, VolumeDim, TargetFrame>& inverse_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric);

template <size_t VolumeDim, typename TargetFrame>
Scalar<DataVector> area_element(
    const InverseJacobian<DataVector, VolumeDim, Frame::ElementLogical,
                          TargetFrame>& inverse_jacobian_face,
    const Scalar<DataVector>& inverse_jacobian_determinant_face,
    const Direction<VolumeDim>& direction,
    const tnsr::II<DataVector, VolumeDim, TargetFrame>& inverse_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric);

template <size_t VolumeDim, typename TargetFrame>
Scalar<DataVector> area_element(
    const InverseJacobian<DataVector, VolumeDim, Frame::ElementLogical,
                          TargetFrame>& inverse_jacobian_face,
    const Direction<VolumeDim>& direction,
    const tnsr::II<DataVector, VolumeDim, TargetFrame>& inverse_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric);
/// @}
