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

template <typename... Ts>
std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
combine_functions_of_time(const std::vector<std::string>& time_dep_names,
                          Ts... functions_of_time_pack) {
  static_assert(sizeof...(Ts) > 0,
                "Must have at least one set of function of times to combine. "
                "We must have one "
                "entry per TimeDependence being composed.");

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  size_t counter_into_dep_names = 0;

  const auto insert_into_map = [&counter_into_dep_names, &functions_of_time,
                                &time_dep_names](
                                   std::unordered_map<
                                       std::string,
                                       std::unique_ptr<domain::FunctionsOfTime::
                                                           FunctionOfTime>>&&
                                       functions_of_time_single_map) {
    for (auto& name_and_func : functions_of_time_single_map) {
      if (UNLIKELY(functions_of_time.count(name_and_func.first) != 0)) {
        ERROR("Inserting already known function '"
              << name_and_func.first
              << "' into composed functions of time. You must change the name "
                 "of one the functions of time being composed. This is done as "
                 "an option to the time dependence in the input file. The "
                 "corresponding time dependence with the duplicate name is '"
              << time_dep_names[counter_into_dep_names] << "'.");
      }
      functions_of_time.insert(std::move(name_and_func));
    }
    ++counter_into_dep_names;
  };
  (void)insert_into_map;  // Silence compiler warnings
  EXPAND_PACK_LEFT_TO_RIGHT(insert_into_map(std::move(functions_of_time_pack)));
  return functions_of_time;
}
}  // namespace detail

template <typename TimeDependenceCompTag0, typename... TimeDependenceCompTags>
Composition<TimeDependenceCompTag0, TimeDependenceCompTags...>::Composition(
    tmpl::type_from<TimeDependenceCompTag0> first_time_dep,
    tmpl::type_from<TimeDependenceCompTags>... rest_time_dep)
    : coord_map_(detail::combine_coord_maps(
          dynamic_cast<typename TimeDependenceCompTag0::time_dependence&>(
              *first_time_dep)
              .map_for_composition(),
          dynamic_cast<typename TimeDependenceCompTags::time_dependence&>(
              *rest_time_dep)
              .map_for_composition()...)),
      functions_of_time_(detail::combine_functions_of_time(
          {db::tag_name<TimeDependenceCompTag0>(),
           db::tag_name<TimeDependenceCompTags>()...},
          std::move(first_time_dep->functions_of_time()),
          std::move(rest_time_dep->functions_of_time())...)) {}

template <typename TimeDependenceCompTag0, typename... TimeDependenceCompTags>
Composition<TimeDependenceCompTag0, TimeDependenceCompTags...>::Composition(
    CoordMap coord_map,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time)
    : coord_map_(std::move(coord_map)) {
  functions_of_time_ = clone_unique_ptrs(functions_of_time);
}

template <typename TimeDependenceCompTag0, typename... TimeDependenceCompTags>
auto Composition<TimeDependenceCompTag0, TimeDependenceCompTags...>::get_clone()
    const -> std::unique_ptr<TimeDependence<mesh_dim>> {
  return std::make_unique<Composition>(coord_map_, functions_of_time_);
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
auto Composition<TimeDependenceCompTag0,
                 TimeDependenceCompTags...>::functions_of_time() const
    -> std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> {
  return clone_unique_ptrs(functions_of_time_);
}
}  // namespace time_dependence
}  // namespace creators
}  // namespace domain
