// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Utilities/Array.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup TensorGroup
/// \brief Combines a time component of a tensor with spatial components to
/// produce a spacetime tensor.
///
/// \details Combines a time component of a tensor with spatial components to
/// produce a spacetime tensor. Specifically, the components of the result
/// are views to the inputs. Can do so for a tensor of any rank, but
/// requires that the new index is the first index of the resulting tensor,
/// replacing the position of the spatial index in the input spatial tensor.
/// For instance, it may combine \f$ \phi \f$ with \f$ A^i \f$ into
/// \f$ A^a = \left(\phi, A^i\right) \f$, or it may combine \f$ A^a{}_b{}_c \f$
/// with \f$ B_i{}^a{}_b{}_c \f$ into
/// \f$ C_a{}^b{}_c{}_d = \left(A^b{}_c{}_d, B_i{}^b{}_c{}_d\right)\f$,
/// but it may not combine \f$ A^i{}_a \f$ with \f$ B^i{}_j{}_a \f$ to produce
/// a tensor of the form \f$ C^i{}_a{}_b \f$.
///
/// \tparam SpatialDim the number of spatial dimensions in the input and output
///         tensors
/// \tparam Ul whether the new index is covariant or contravariant (must match
///         that of the spatial index of the input spatial tensor)
/// \tparam Frame the frame of the new spacetime index (must match that of the
///         spatial index of the input spatial tensor)
template <size_t SpatialDim, UpLo Ul, typename Frame, typename DataType,
          typename SymmList, typename IndexList>
void combine_spacetime_view(
    gsl::not_null<TensorMetafunctions::prepend_spacetime_index<
        Tensor<DataType, SymmList, IndexList>, SpatialDim, Ul, Frame>*>
        spacetime_tensor,
    const Tensor<DataType, SymmList, IndexList>& time_tensor,
    const TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, SpatialDim, Ul, Frame>&
        spatial_tensor) {
  for (size_t storage_index = 0;
       storage_index < Tensor<DataVector, SymmList, IndexList>::size();
       ++storage_index) {
    const auto u_multi_index =
        Tensor<DataVector, SymmList,
               IndexList>::structure::get_canonical_tensor_index(storage_index);
    if constexpr (std::is_same_v<DataType, DataVector>) {
      const auto dtu_multi_index = prepend(u_multi_index, 0_st);
      make_const_view(
          make_not_null(&std::as_const(spacetime_tensor->get(dtu_multi_index))),
          time_tensor.get(u_multi_index), 0,
          time_tensor.get(u_multi_index).size());
      for (size_t i = 0; i < SpatialDim; i++) {
        const auto du_multi_index = prepend(u_multi_index, i + 1);
        const auto diu_multi_index = prepend(u_multi_index, i);
        make_const_view(make_not_null(&std::as_const(
                            spacetime_tensor->get(du_multi_index))),
                        spatial_tensor.get(diu_multi_index), 0,
                        spatial_tensor.get(diu_multi_index).size());
      }
    } else {
      const auto dtu_multi_index = prepend(u_multi_index, 0_st);
      spacetime_tensor->get(dtu_multi_index) = time_tensor.get(u_multi_index);
      for (size_t i = 0; i < SpatialDim; ++i) {
        const auto du_multi_index = prepend(u_multi_index, i + 1);
        const auto diu_multi_index = prepend(u_multi_index, i);
        spacetime_tensor->get(du_multi_index) =
            spatial_tensor.get(diu_multi_index);
      }
    }
  }
}
