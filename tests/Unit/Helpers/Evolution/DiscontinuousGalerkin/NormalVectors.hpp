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

// Takes in a spatial metric, or inverse spatial metric, with random components
// in the range [-1,1]. Adjusts the metric (or inverse) by scaling all
// components to lie in [0,0.01] and then adding 1 to the diagonal.
// This is to give an inverse spatial metric of the form:
//  \delta^{ij} + small^{ij}
// This is done to give a physically reasonable spatial metric (or inverse)
template <typename Index>
void adjust_spatial_metric_or_inverse(
    const gsl::not_null<
        Tensor<DataVector, Symmetry<1, 1>, index_list<Index, Index>>*>
        spatial_metric_or_inverse) {
  for (size_t i = 0; i < Index::dim; ++i) {
    for (size_t j = i; j < Index::dim; ++j) {
      spatial_metric_or_inverse->get(i, j) += 1.0;
      spatial_metric_or_inverse->get(i, j) *= 0.005;
    }
  }
  for (size_t i = 0; i < Index::dim; ++i) {
    spatial_metric_or_inverse->get(i, i) += 1.0;
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
