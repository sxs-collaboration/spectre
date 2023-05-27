// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FlatLogicalMetric.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace domain {

template <size_t Dim>
void flat_logical_metric(
    const gsl::not_null<tnsr::ii<DataVector, Dim, Frame::ElementLogical>*>
        result,
    const Jacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>&
        jacobian) {
  destructive_resize_components(result, jacobian.begin()->size());
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      result->get(i, j) = 0.;
      for (size_t k = 0; k < Dim; ++k) {
        result->get(i, j) += jacobian.get(k, i) * jacobian.get(k, j);
      }
    }
  }
}

namespace Tags {

template <size_t Dim>
void FlatLogicalMetricCompute<Dim>::function(
    const gsl::not_null<tnsr::ii<DataVector, Dim, Frame::ElementLogical>*>
        result,
    const ::InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian) {
  // We have the inverse Jacobian available, so just do a numerical inverse to
  // compute the Jacobian. That's easier than computing the Jacobian from
  // time-dependent maps here, though if the Jacobian is available then it's
  // of course better to use it here directly.
  const auto jacobian = determinant_and_inverse(inv_jacobian).second;
  flat_logical_metric(result, jacobian);
}

template class FlatLogicalMetricCompute<1>;
template class FlatLogicalMetricCompute<2>;
template class FlatLogicalMetricCompute<3>;

}  // namespace Tags
}  // namespace domain
