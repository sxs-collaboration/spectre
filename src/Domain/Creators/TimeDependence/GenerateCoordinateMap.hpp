// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain::creators::time_dependence::detail {
template <typename SourceFrame, typename TargetFrame, typename MapsList>
struct generate_coordinate_map;

template <typename SourceFrame, typename TargetFrame, typename... Maps>
struct generate_coordinate_map<SourceFrame, TargetFrame, tmpl::list<Maps...>> {
  using type = domain::CoordinateMap<SourceFrame, TargetFrame, Maps...>;
};

template <typename SourceFrame, typename TargetFrame, typename MapsList>
using generate_coordinate_map_t =
    typename generate_coordinate_map<SourceFrame, TargetFrame, MapsList>::type;
}  // namespace domain::creators::time_dependence::detail
