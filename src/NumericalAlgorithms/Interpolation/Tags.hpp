// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <deque>
#include <string>
#include <unordered_set>

#include "DataStructures/DataBox/DataBoxTag.hpp"

/// \cond
template <size_t Dim, typename Frame>
class Domain;
/// \endcond

namespace intrp {

/// Tags for items held in the DataBox of InterpolationTarget or Interpolator.
namespace Tags {

/// Keeps track of which points have been filled with interpolated data.
struct IndicesOfFilledInterpPoints : db::SimpleTag {
  static std::string name() noexcept { return "IndicesOfFilledInterpPoints"; }
  using type = std::unordered_set<size_t>;
};

/// TemporalIds on which to interpolate.
template <typename Metavariables>
struct TemporalIds : db::SimpleTag {
  using type = std::deque<typename Metavariables::temporal_id>;
  static std::string name() noexcept { return "TemporalIds"; }
};

}  // namespace Tags
}  // namespace intrp
