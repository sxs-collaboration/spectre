// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"

namespace TestHelpers::evolution::dg::detail {
CREATE_HAS_TYPE_ALIAS(inverse_spatial_metric_tag)
CREATE_HAS_TYPE_ALIAS_V(inverse_spatial_metric_tag)

template <bool HasInverseSpatialMetricTag = false>
struct inverse_spatial_metric_tag {
  template <typename System>
  using f = tmpl::list<>;
};

template <>
struct inverse_spatial_metric_tag<true> {
  template <typename System>
  using f = typename System::inverse_spatial_metric_tag;
};

// On input `inv_spatial_metric` is expected to have components on the interval
// [0, 1]. The components are rescaled by 0.01, and 1 is added to the diagonal.
// This is to give an inverse spatial metric of the form:
//  \delta^{ij} + small^{ij}
// This is done to give a physically reasonable inverse spatial metric
template <size_t Dim>
void adjust_inverse_spatial_metric(
    const gsl::not_null<tnsr::II<DataVector, Dim>*> inv_spatial_metric) {
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      inv_spatial_metric->get(i, j) *= 0.01;
    }
  }
  for (size_t i = 0; i < Dim; ++i) {
    inv_spatial_metric->get(i, i) += 1.0;
  }
}

// On input the `unit_normal_covector` is the unnormalized normal covector. On
// output `unit_normal_covector` is the normalized (and hence actually unit)
// normal covector, and `unit_normal_vector` is the unit normal vector. The
// inverse spatial metric is used for computing the magnitude of the
// unnormalized normal vector.
template <size_t Dim>
void normalize_vector_and_covector(
    const gsl::not_null<tnsr::i<DataVector, Dim>*> unit_normal_covector,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> unit_normal_vector,
    const tnsr::II<DataVector, Dim>& inv_spatial_metric) {
  for (size_t i = 0; i < Dim; ++i) {
    unit_normal_vector->get(i) =
        inv_spatial_metric.get(i, 0) * get<0>(*unit_normal_covector);
    for (size_t j = 1; j < Dim; ++j) {
      unit_normal_vector->get(i) +=
          inv_spatial_metric.get(i, j) * unit_normal_covector->get(j);
    }
  }

  const DataVector normal_magnitude =
      sqrt(get(dot_product(*unit_normal_covector, *unit_normal_vector)));
  for (auto& t : *unit_normal_covector) {
    t /= normal_magnitude;
  }
  for (auto& t : *unit_normal_vector) {
    t /= normal_magnitude;
  }
}
}  // namespace TestHelpers::evolution::dg::detail
