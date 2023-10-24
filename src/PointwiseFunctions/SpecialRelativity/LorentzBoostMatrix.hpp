// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

/// \endcond

/// \ingroup SpecialRelativityGroup
/// Holds functions related to special relativity.
namespace sr {
/// @{
/*!
 * \ingroup SpecialRelativityGroup
 * \brief Computes the matrix for a Lorentz boost from a single
 * velocity vector (i.e., not a velocity field).
 *
 * \details Given a spatial velocity vector \f$v^i\f$ (with \f$c=1\f$),
 * compute the matrix \f$\Lambda^{a}{}_{\bar{a}}\f$ for a Lorentz boost with
 * that velocity [e.g. Eq. (2.38) of \cite ThorneBlandford2017]:
 *
 * \f{align}{
 * \Lambda^t{}_{\bar{t}} &= \gamma, \\
 * \Lambda^t{}_{\bar{i}} = \Lambda^i{}_{\bar{t}} &= \gamma v^i, \\
 * \Lambda^i{}_{\bar{j}} = \Lambda^j{}_{\bar{i}} &= [(\gamma - 1)/v^2] v^i v^j
 *                                              + \delta^{ij}.
 * \f}
 *
 * Here \f$v = \sqrt{\delta_{ij} v^i v^j}\f$, \f$\gamma = 1/\sqrt{1-v^2}\f$,
 * and \f$\delta^{ij}\f$ is the Kronecker delta. Note that this matrix boosts
 * a one-form from the unbarred to the barred frame, and its inverse
 * (obtained via \f$v \rightarrow -v\f$) boosts a vector from the barred to
 * the unbarred frame.
 *
 * Note that while the Lorentz boost matrix is symmetric, the returned
 * boost matrix is of type `tnsr::Ab`, because `Tensor` does not support
 * symmetric tensors unless both indices have the same valence.
 */
template <size_t SpatialDim>
tnsr::Ab<double, SpatialDim, Frame::NoFrame> lorentz_boost_matrix(
    const tnsr::I<double, SpatialDim, Frame::NoFrame>& velocity);

template <size_t SpatialDim>
void lorentz_boost_matrix(
    gsl::not_null<tnsr::Ab<double, SpatialDim, Frame::NoFrame>*> boost_matrix,
    const tnsr::I<double, SpatialDim, Frame::NoFrame>& velocity);

template <size_t SpatialDim>
tnsr::Ab<double, SpatialDim, Frame::NoFrame> lorentz_boost_matrix(
    const std::array<double, SpatialDim>& velocity);
/// @}

template <typename DataType, size_t SpatialDim, typename SourceFrame,
          typename TargetFrame, typename DataTypeComponent0>
void lorentz_boost(
    const gsl::not_null<tnsr::I<DataType, SpatialDim, TargetFrame>*> result,
    const tnsr::I<DataType, SpatialDim, SourceFrame>& vector,
    const DataTypeComponent0& vector_component_0,
    const std::array<double, SpatialDim>& velocity) {
  if (velocity == make_array<SpatialDim>(0.)) {
    *result = vector;
    return;
  }
  const auto boost_matrix = lorentz_boost_matrix(velocity);
  for (size_t i = 0; i < SpatialDim; ++i) {
    result->get(i) = boost_matrix.get(i + 1, 0) * vector_component_0;
    for (size_t j = 0; j < SpatialDim; ++j) {
      result->get(i) += boost_matrix.get(i + 1, j + 1) * vector.get(j);
    }
  }
}

template <typename DataType, size_t SpatialDim, typename SourceFrame,
          typename TargetFrame>
void lorentz_boost(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, TargetFrame>*> result,
    const tnsr::a<DataType, SpatialDim, SourceFrame>& vector,
    const std::array<double, SpatialDim>& velocity) {
  if (velocity == make_array<SpatialDim>(0.)) {
    *result = vector;
    return;
  }
  const auto boost_matrix = lorentz_boost_matrix(velocity);
  for (size_t i = 0; i < SpatialDim + 1; ++i) {
    result->get(i) = 0.;
    for (size_t j = 0; j < SpatialDim + 1; ++j) {
      result->get(i) += boost_matrix.get(i, j) * vector.get(j);
    }
  }
}

template <typename DataType, size_t SpatialDim, typename SourceFrame,
          typename TargetFrame>
void lorentz_boost(
    const gsl::not_null<tnsr::ab<DataType, SpatialDim, TargetFrame>*> result,
    const tnsr::ab<DataType, SpatialDim, SourceFrame>& tensor,
    const std::array<double, SpatialDim>& velocity_first_index,
    const std::array<double, SpatialDim>& velocity_second_index) {
  if (velocity_first_index == make_array<SpatialDim>(0.) and
      velocity_second_index == make_array<SpatialDim>(0.)) {
    *result = tensor;
    return;
  }
  const auto boost_matrix_1 = lorentz_boost_matrix(velocity_first_index);
  const auto boost_matrix_2 = lorentz_boost_matrix(velocity_second_index);
  for (size_t i = 0; i < SpatialDim + 1; ++i) {
    for (size_t k = 0; k < SpatialDim + 1; ++k) {
      result->get(i, k) = 0.;
      for (size_t j = 0; j < SpatialDim + 1; ++j) {
        for (size_t l = 0; l < SpatialDim + 1; ++l) {
          result->get(i, k) += boost_matrix_1.get(i, j) *
                               boost_matrix_2.get(k, l) * tensor.get(j, l);
        }
      }
    }
  }
}

}  // namespace sr
