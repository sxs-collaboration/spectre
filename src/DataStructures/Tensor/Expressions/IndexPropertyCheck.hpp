// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/Expressions/TimeIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
namespace detail {
/// @{
/// \brief Helper struct for checking that one tensor's index is either a
/// spacetime index where a concrete time index has been used or can be
/// assigned to, added to, or subtracted from its corresponding index in another
/// tensor
///
/// \details
/// Indices in one tensor correspond to those in another that use the same
/// generic index, such as `ti_a`. For it to be possible to add, subtract, or
/// assign one index to another, this checks that the following is true for the
/// index and its corresponding index in another tensor:
/// - has the same valence (`UpLo`)
/// - has the same `Frame` type
/// - has the same number of spatial dimensions (allowing for expressions that
///   use generic spatial indices for spacetime indices on either side)
///
/// \tparam IndexList1 the first tensor's \ref SpacetimeIndex "TensorIndexType"
/// list
/// \tparam IndexList2 the second tensor's \ref SpacetimeIndex "TensorIndexType"
/// list
/// \tparam TensorIndexList1 the first tensor's generic index list
/// \tparam TensorIndexList2 the second tensor's generic index list
/// \tparam CurrentTensorIndex1 the current generic index of the first tensor
/// that is being checked, e.g. the type of `ti_a`
/// \tparam Iteration the position of the current index of the first tensor
/// being checked, e.g. the position of `ti_a` in the first tensor
template <typename IndexList1, typename IndexList2, typename TensorIndexList1,
          typename TensorIndexList2, typename CurrentTensorIndex1,
          typename Iteration>
struct IndexPropertyCheckImpl {
  using index1 = tmpl::at<IndexList1, Iteration>;
  using index2 =
      tmpl::at<IndexList2,
               tmpl::index_of<TensorIndexList2, CurrentTensorIndex1>>;

  using type = std::bool_constant<
      index1::ul == index2::ul and
      std::is_same_v<typename index1::Frame, typename index2::Frame> and
      ((index1::index_type == index2::index_type and
        index1::dim == index2::dim) or
       (index1::index_type == IndexType::Spacetime and
        index1::dim == index2::dim + 1) or
       (index2::index_type == IndexType::Spacetime and
        index1::dim + 1 == index2::dim))>;
};

template <typename IndexList1, typename IndexList2, typename TensorIndexList1,
          typename TensorIndexList2, typename Iteration>
struct IndexPropertyCheckImpl<IndexList1, IndexList2, TensorIndexList1,
                              TensorIndexList2, std::decay_t<decltype(ti_T)>,
                              Iteration> {
  using index1 = tmpl::at<IndexList1, Iteration>;
  using type = std::bool_constant<index1::index_type == IndexType::Spacetime>;
};

template <typename IndexList1, typename IndexList2, typename TensorIndexList1,
          typename TensorIndexList2, typename Iteration>
struct IndexPropertyCheckImpl<IndexList1, IndexList2, TensorIndexList1,
                              TensorIndexList2, std::decay_t<decltype(ti_t)>,
                              Iteration> {
  using index1 = tmpl::at<IndexList1, Iteration>;
  using type = std::bool_constant<index1::index_type == IndexType::Spacetime>;
};
/// @}

/// \brief Helper struct for checking that one tensor's indices can be
/// mathematically assigned to, added to, or subtracted from their
/// corresponding indices in another tensor
///
/// \details
/// This struct checks that:
/// (1) The shared generic indices between the two index lists can be
/// mathematically assigned to, added to, or subtracted from one another
/// (2) Any non-shared indices are spacetime indices where a concrete time index
/// has been used
///
/// This struct checks that (2) is true for the second tensor's time indices and
/// calls `IndexPropertyCheckImpl` to check that (2) is true for the first
/// tensor's time indices and that (1) is true. To see more details regarding
/// how (1) is checked, see `IndexPropertyCheckImpl`.
///
/// \tparam IndexList1 the first tensor's \ref SpacetimeIndex "TensorIndexType"
/// list
/// \tparam IndexList2 the second tensor's \ref SpacetimeIndex "TensorIndexType"
/// list
/// \tparam TensorIndexList1 the first tensor's generic index list
/// \tparam TensorIndexList2 the second tensor's generic index list
template <typename IndexList1, typename IndexList2, typename TensorIndexList1,
          typename TensorIndexList2>
struct IndexPropertyCheckHelper;

template <typename IndexList1, typename... RhsIndices,
          typename TensorIndexList1, typename... RhsTensorIndices>
struct IndexPropertyCheckHelper<IndexList1, tmpl::list<RhsIndices...>,
                                TensorIndexList1,
                                tmpl::list<RhsTensorIndices...>> {
  static constexpr bool value =
      // Check that second tensor's concrete time indices are used with
      // spacetime indices
      (... and ((not tt::is_time_index<RhsTensorIndices>::value) or
                (tt::is_time_index<RhsTensorIndices>::value and
                 RhsIndices::index_type == IndexType::Spacetime))) and
      // Check that:
      // - the first tensor's concrete time indices are used with spacetime
      // indices
      // - shared generic indices between the two tensors can be mathematically
      // assigned to, added to, or subtracted from one another
      (tmpl::enumerated_fold<
          TensorIndexList1, tmpl::bool_<true>,
          tmpl::and_<
              tmpl::_state,
              IndexPropertyCheckImpl<tmpl::pin<IndexList1>,
                                     tmpl::pin<tmpl::list<RhsIndices...>>,
                                     tmpl::pin<TensorIndexList1>,
                                     tmpl::pin<tmpl::list<RhsTensorIndices...>>,
                                     tmpl::_element, tmpl::_3>>,
          tmpl::size_t<0>>::value);
};

/// \brief Check that one tensor's indices can be mathematically assigned to,
/// added to, or subtracted from their corresponding indices in another tensor
///
/// \details
/// For more details on what is checked, see `IndexPropertyCheckImpl` followed
/// by `IndexPropertyCheckHelper`
///
/// \tparam IndexList1 the first tensor's \ref SpacetimeIndex "TensorIndexType"
/// list
/// \tparam IndexList2 the second tensor's \ref SpacetimeIndex "TensorIndexType"
/// list
/// \tparam TensorIndexList1 the first tensor's generic index list
/// \tparam TensorIndexList2 the second tensor's generic index list
template <typename IndexList1, typename IndexList2, typename TensorIndexList1,
          typename TensorIndexList2>
using IndexPropertyCheck =
    IndexPropertyCheckHelper<IndexList1, IndexList2, TensorIndexList1,
                             TensorIndexList2>;
}  // namespace detail
}  // namespace TensorExpressions
