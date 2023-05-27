// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace domain {

/*!
 * \brief The "flat logical metric" $\sum_k \frac{\partial x^k}{\partial
 * \xi^\hat{i}} \frac{\partial x^k}{\partial \xi^\hat{j}}$, which is the flat
 * spatial metric in element-logical coordinates.
 *
 * We define the "flat logical metric" to be the matrix of inner products of the
 * $\hat{i}$-th logical coordinate basis vector $\partial_{\xi^\hat{i}}$ with
 * the $\hat{j}$-th logical coordinate basis vector $\partial_{\xi^\hat{j}}$,
 * where the inner product is taken assuming a flat spatial metric. When
 * expressed in the ("inertial") $x$-coordinate system, each basis vector is a
 * column of the Jacobian.
 */
template <size_t Dim>
void flat_logical_metric(
    const gsl::not_null<tnsr::ii<DataVector, Dim, Frame::ElementLogical>*>
        result,
    const Jacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>&
        jacobian);

namespace Tags {
/// The flat metric in element-logical coordinates
///
/// \see domain::flat_logical_metric
template <size_t Dim>
struct FlatLogicalMetric : db::SimpleTag {
  using type = tnsr::ii<DataVector, Dim, Frame::ElementLogical>;
};

/// Compute the flat metric in element-logical coordinates from the inverse
/// Jacobian.
///
/// \see domain::flat_logical_metric
template <size_t Dim>
struct FlatLogicalMetricCompute : FlatLogicalMetric<Dim>, db::ComputeTag {
  using base = FlatLogicalMetric<Dim>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                               Frame::Inertial>>;
  static void function(
      const gsl::not_null<tnsr::ii<DataVector, Dim, Frame::ElementLogical>*>
          result,
      const ::InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                              Frame::Inertial>& inv_jacobian);
};

}  // namespace Tags
}  // namespace domain
