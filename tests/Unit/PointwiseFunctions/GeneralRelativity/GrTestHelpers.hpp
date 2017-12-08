// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions useful for testing general relativity

#pragma once

#include <boost/range/combine.hpp>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "tests/Unit/TestHelpers.hpp"

template <typename Symmetry, typename IndexList>
void check_tensor_doubles_equals_tensor_datavectors(
    const Tensor<DataVector, Symmetry, IndexList>& tensor_dv,
    const Tensor<double, Symmetry, IndexList>& tensor_double) {
  const size_t n_pts = tensor_dv.begin()->size();
  for (decltype(auto) datavector_and_double_components :
       boost::combine(tensor_dv, tensor_double)) {
    for (size_t s = 0; s < n_pts; ++s) {
      CHECK(boost::get<0>(datavector_and_double_components)[s] ==
            boost::get<1>(datavector_and_double_components));
    }
  }
}

template <typename DataType>
Scalar<DataType> make_lapse(const DataType& used_for_size);

template <size_t SpatialDim, typename DataType>
tnsr::I<DataType, SpatialDim> make_shift(const DataType& used_for_size);

template <size_t SpatialDim, typename DataType>
tnsr::ii<DataType, SpatialDim> make_spatial_metric(
    const DataType& used_for_size);

template <size_t SpatialDim, typename DataType>
tnsr::II<DataType, SpatialDim> make_inverse_spatial_metric(
    const DataType& used_for_size);

template <typename DataType>
Scalar<DataType> make_dt_lapse(const DataType& used_for_size);

template <size_t SpatialDim, typename DataType>
tnsr::I<DataType, SpatialDim> make_dt_shift(const DataType& used_for_size);

template <size_t SpatialDim, typename DataType>
tnsr::ii<DataType, SpatialDim> make_dt_spatial_metric(
    const DataType& used_for_size);

template <size_t SpatialDim, typename DataType>
tnsr::i<DataType, SpatialDim> make_deriv_lapse(const DataType& used_for_size);

template <size_t SpatialDim, typename DataType>
tnsr::iJ<DataType, SpatialDim> make_deriv_shift(const DataType& used_for_size);

template <size_t SpatialDim, typename DataType>
tnsr::ijj<DataType, SpatialDim> make_deriv_spatial_metric(
    const DataType& used_for_size);

template <size_t SpatialDim, typename DataType>
tnsr::Ijj<DataType, SpatialDim> make_spatial_christoffel_second_kind(
    const DataType& used_for_size);

template <size_t SpatialDim, typename DataType>
tnsr::abb<DataType, SpatialDim> make_spacetime_deriv_spacetime_metric(
    const DataType& used_for_size);

template <size_t SpatialDim, typename DataType>
tnsr::aa<DataType, SpatialDim> make_dt_spacetime_metric(
    const DataType& used_for_size);

template <size_t SpatialDim, typename DataType>
tnsr::iaa<DataType, SpatialDim> make_spatial_deriv_spacetime_metric(
    const DataType& used_for_size);

template <size_t SpatialDim, typename DataType>
tnsr::Abb<DataType, SpatialDim> make_spacetime_christoffel_second_kind(
    const DataType& used_for_size);

template <size_t SpatialDim, typename DataType>
tnsr::aa<DataType, SpatialDim> make_spacetime_metric(
    const DataType& used_for_size);

template <size_t SpatialDim, typename DataType>
tnsr::AA<DataType, SpatialDim> make_inverse_spacetime_metric(
    const DataType& used_for_size);
