// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
namespace Frame {
struct Grid;
struct Inertial;
}  // namespace Frame
/// \endcond

namespace domain {
namespace creators {
namespace time_dependence {
namespace detail {
template <typename MapsList>
struct generate_coordinate_map;

template <typename... Maps>
struct generate_coordinate_map<tmpl::list<Maps...>> {
  using type = domain::CoordinateMap<Frame::Grid, Frame::Inertial, Maps...>;
};

template <typename MapsList>
using generate_coordinate_map_t =
    typename generate_coordinate_map<MapsList>::type;
}  // namespace detail
}  // namespace time_dependence
}  // namespace creators
}  // namespace domain
