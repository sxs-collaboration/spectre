// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/Creators/TimeDependence/Composition.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/TagName.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/TMPL.hpp"

namespace domain {
namespace creators {
namespace time_dependence {
namespace detail {
template <typename... MapsSoFar>
auto combine_coord_maps(
    CoordinateMap<Frame::Grid, Frame::Inertial, MapsSoFar...>
        coord_map_so_far) {
  return coord_map_so_far;
}

template <typename... MapsSoFar, typename... NextMaps, typename... RestMaps>
auto combine_coord_maps(
    CoordinateMap<Frame::Grid, Frame::Inertial, MapsSoFar...> coord_map_so_far,
    CoordinateMap<Frame::Grid, Frame::Inertial, NextMaps...> next_map,
    RestMaps... rest_maps) {
  return combine_coord_maps(
      domain::push_back(std::move(coord_map_so_far), std::move(next_map)),
      std::move(rest_maps)...);
}
}  // namespace detail

template <typename TimeDependenceCompTag0, typename... TimeDependenceCompTags>
Composition<TimeDependenceCompTag0, TimeDependenceCompTags...>::Composition(
    tmpl::type_from<TimeDependenceCompTag0> first_time_dep,
    tmpl::type_from<TimeDependenceCompTags>... rest_time_dep)
    : coord_map_(
          detail::combine_coord_maps(first_time_dep.map_for_composition(),
                                     rest_time_dep.map_for_composition()...)) {
  time_deps_.push_back(first_time_dep.get_clone());
  time_deps_.push_back(rest_time_dep.get_clone()...);
}

template <typename TimeDependenceCompTag0, typename... TimeDependenceCompTags>
Composition<TimeDependenceCompTag0, TimeDependenceCompTags...>::Composition(
    CoordMap coord_map,
    const std::vector<
        std::unique_ptr<TimeDependence<TimeDependenceCompTag0::mesh_dim>>>&
        time_deps)
    : coord_map_(coord_map), time_deps_(clone_unique_ptrs(time_deps)) {}

template <typename TimeDependenceCompTag0, typename... TimeDependenceCompTags>
auto Composition<TimeDependenceCompTag0, TimeDependenceCompTags...>::get_clone()
    const -> std::unique_ptr<TimeDependence<mesh_dim>> {
  return std::make_unique<Composition>(coord_map_, time_deps_);
}

template <typename TimeDependenceCompTag0, typename... TimeDependenceCompTags>
auto Composition<TimeDependenceCompTag0, TimeDependenceCompTags...>::block_maps(
    const size_t number_of_blocks) const
    -> std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, mesh_dim>>> {
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, mesh_dim>>>
      result{number_of_blocks};
  result[0] = std::make_unique<CoordMap>(coord_map_);
  for (size_t i = 1; i < number_of_blocks; ++i) {
    result[i] = result[0]->get_clone();
  }
  return result;
}

template <typename TimeDependenceCompTag0, typename... TimeDependenceCompTags>
auto Composition<TimeDependenceCompTag0, TimeDependenceCompTags...>::
    functions_of_time(const std::unordered_map<std::string, double>&
                          initial_expiration_times) const
    -> std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  for (auto& time_dep : time_deps_) {
    functions_of_time.merge(
        time_dep->functions_of_time(initial_expiration_times));
  }

  return functions_of_time;
}
}  // namespace time_dependence
}  // namespace creators
}  // namespace domain
