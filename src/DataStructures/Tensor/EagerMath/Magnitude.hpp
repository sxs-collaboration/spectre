// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \ingroup TensorGroup
 * \brief Compute the Euclidean magnitude of a rank-1 tensor
 *
 * \details
 * Computes the square root of the sum of the squares of the components of
 * the rank-1 tensor.
 */
template <typename DataType, typename Index>
Scalar<DataType> magnitude(
    const Tensor<DataType, Symmetry<1>, index_list<Index>>& vector) noexcept {
  return Scalar<DataType>{sqrt(get(dot_product(vector, vector)))};
}

/*!
 * \ingroup TensorGroup
 * \brief Compute the magnitude of a rank-1 tensor
 *
 * \details
 * Returns the square root of the input tensor contracted twice with the given
 * metric.
 */
template <typename DataType, typename Index>
Scalar<DataType> magnitude(
    const Tensor<DataType, Symmetry<1>, index_list<Index>>& vector,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index>,
                            change_index_up_lo<Index>>>&
        metric) noexcept {
  return Scalar<DataType>{sqrt(get(dot_product(vector, vector, metric)))};
}

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup DataStructuresGroup
/// The Euclidean magnitude of a (co)vector
template <typename Tag>
struct EuclideanMagnitude : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "EuclideanMagnitude";
  static constexpr Scalar<DataVector> (*function)(const db::item_type<Tag>&) =
      magnitude;
  using argument_tags = tmpl::list<Tag>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup DataStructuresGroup
/// The (co)vector represented by Tag normalized by its magnitude from
/// MagnitudeTag.
template <typename Tag, typename MagnitudeTag>
struct Normalized : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "Normalized";
  static constexpr auto function(
      db::item_type<Tag> vector,
      const db::item_type<MagnitudeTag>& magnitude) noexcept {
    for (size_t d = 0; d < vector.index_dim(0); ++d) {
      vector.get(d) /= get(magnitude);
    }
    return vector;
  }
  using argument_tags = tmpl::list<Tag, MagnitudeTag>;
};
}  // namespace Tags
