// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"

#include <array>
#include <boost/none.hpp>
#include <boost/optional.hpp>
#include <cstddef>
#include <functional>
#include <pup.h>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMapHelpers.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace domain {
namespace CoordinateMaps {
namespace TimeDependent {
namespace product_detail {
template <typename T, size_t Size, typename Map1, typename Map2, size_t... Is,
          size_t... Js>
std::array<tt::remove_cvref_wrap_t<T>, Size> apply_map(
    const std::array<T, Size>& coords, const Map1& map1, const Map2& map2,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    std::integer_sequence<size_t, Is...> /*meta*/,
    std::integer_sequence<size_t, Js...> /*meta*/) noexcept {
  using UnwrappedT = tt::remove_cvref_wrap_t<T>;
  return {
      {CoordinateMap_detail::apply_map(
           map1,
           std::array<std::reference_wrapper<const UnwrappedT>, sizeof...(Is)>{
               {coords[Is]...}},
           time, functions_of_time,
           domain::is_map_time_dependent_t<Map1>{})[Is]...,
       CoordinateMap_detail::apply_map(
           map2,
           std::array<std::reference_wrapper<const UnwrappedT>, sizeof...(Js)>{
               {coords[Map1::dim + Js]...}},
           time, functions_of_time,
           domain::is_map_time_dependent_t<Map2>{})[Js]...}};
}

template <size_t Size, typename Map1, typename Map2, size_t... Is, size_t... Js>
boost::optional<std::array<double, Size>> apply_inverse(
    const std::array<double, Size>& coords, const Map1& map1, const Map2& map2,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    std::integer_sequence<size_t, Is...> /*meta*/,
    std::integer_sequence<size_t, Js...> /*meta*/) noexcept {
  auto map1_func = CoordinateMap_detail::apply_inverse_map(
      map1, std::array<double, sizeof...(Is)>{{coords[Is]...}}, time,
      functions_of_time, domain::is_map_time_dependent_t<Map1>{});
  auto map2_func = CoordinateMap_detail::apply_inverse_map(
      map2, std::array<double, sizeof...(Js)>{{coords[Map1::dim + Js]...}},
      time, functions_of_time, domain::is_map_time_dependent_t<Map2>{});
  if (map1_func and map2_func) {
    return {{{map1_func.get()[Is]..., map2_func.get()[Js]...}}};
  } else {
    return boost::none;
  }
}

template <typename T, size_t Size, typename Map1, typename Map2, size_t... Is,
          size_t... Js>
std::array<tt::remove_cvref_wrap_t<T>, Size> apply_frame_velocity(
    const std::array<T, Size>& coords, const Map1& map1, const Map2& map2,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    std::integer_sequence<size_t, Is...> /*meta*/,
    std::integer_sequence<size_t, Js...> /*meta*/) noexcept {
  using UnwrappedT = tt::remove_cvref_wrap_t<T>;
  return {
      {domain::CoordinateMap_detail::apply_frame_velocity(
           map1,
           std::array<std::reference_wrapper<const UnwrappedT>, sizeof...(Is)>{
               {coords[Is]...}},
           time, functions_of_time,
           domain::is_map_time_dependent_t<Map1>{})[Is]...,
       domain::CoordinateMap_detail::apply_frame_velocity(
           map2,
           std::array<std::reference_wrapper<const UnwrappedT>, sizeof...(Js)>{
               {coords[Map1::dim + Js]...}},
           time, functions_of_time,
           domain::is_map_time_dependent_t<Map2>{})[Js]...}};
}

template <typename T, size_t Size, typename Map1, typename Map2,
          typename Function, size_t... Is, size_t... Js>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, Size, Frame::NoFrame> apply_jac(
    const std::array<T, Size>& source_coords, const Map1& map1,
    const Map2& map2, const Function func,
    std::integer_sequence<size_t, Is...> /*meta*/,
    std::integer_sequence<size_t, Js...> /*meta*/) noexcept {
  using UnwrappedT = tt::remove_cvref_wrap_t<T>;
  auto map1_jac = func(
      std::array<std::reference_wrapper<const UnwrappedT>, sizeof...(Is)>{
          {source_coords[Is]...}},
      map1);
  auto map2_jac = func(
      std::array<std::reference_wrapper<const UnwrappedT>, sizeof...(Js)>{
          {source_coords[Map1::dim + Js]...}},
      map2);
  tnsr::Ij<UnwrappedT, Size, Frame::NoFrame> jac{
      make_with_value<UnwrappedT>(dereference_wrapper(source_coords[0]), 0.0)};
  for (size_t i = 0; i < Map1::dim; ++i) {
    for (size_t j = 0; j < Map1::dim; ++j) {
      jac.get(i, j) = std::move(map1_jac.get(i, j));
    }
  }
  for (size_t i = 0; i < Map2::dim; ++i) {
    for (size_t j = 0; j < Map2::dim; ++j) {
      jac.get(Map1::dim + i, Map1::dim + j) = std::move(map2_jac.get(i, j));
    }
  }
  return jac;
}
}  // namespace product_detail

template <typename Map1, typename Map2>
ProductOf2Maps<Map1, Map2>::ProductOf2Maps(Map1 map1, Map2 map2) noexcept
    : map1_(std::move(map1)), map2_(std::move(map2)) {}

template <typename Map1, typename Map2>
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, ProductOf2Maps<Map1, Map2>::dim>
ProductOf2Maps<Map1, Map2>::operator()(
    const std::array<T, dim>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  return product_detail::apply_map(source_coords, map1_, map2_, time,
                                   functions_of_time,
                                   std::make_index_sequence<Map1::dim>{},
                                   std::make_index_sequence<Map2::dim>{});
}

template <typename Map1, typename Map2>
boost::optional<std::array<double, ProductOf2Maps<Map1, Map2>::dim>>
ProductOf2Maps<Map1, Map2>::inverse(
    const std::array<double, dim>& target_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  return product_detail::apply_inverse(target_coords, map1_, map2_, time,
                                       functions_of_time,
                                       std::make_index_sequence<Map1::dim>{},
                                       std::make_index_sequence<Map2::dim>{});
}

template <typename Map1, typename Map2>
template <typename T>
auto ProductOf2Maps<Map1, Map2>::frame_velocity(
    const std::array<T, dim>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept
    -> std::array<tt::remove_cvref_wrap_t<T>, dim> {
  return product_detail::apply_frame_velocity(
      source_coords, map1_, map2_, time, functions_of_time,
      std::make_index_sequence<Map1::dim>{},
      std::make_index_sequence<Map2::dim>{});
}

template <typename Map1, typename Map2>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, ProductOf2Maps<Map1, Map2>::dim,
         Frame::NoFrame>
ProductOf2Maps<Map1, Map2>::inv_jacobian(
    const std::array<T, dim>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  using UnwrappedT = tt::remove_cvref_wrap_t<T>;
  return product_detail::apply_jac(
      source_coords, map1_, map2_,
      [&time, &functions_of_time](const auto& point, const auto& map) noexcept {
        return CoordinateMap_detail::apply_inverse_jacobian(
            map, point, time, functions_of_time,
            domain::is_jacobian_time_dependent_t<
                std::decay_t<decltype(map)>,
                std::reference_wrapper<const UnwrappedT>>{});
      },
      std::make_index_sequence<Map1::dim>{},
      std::make_index_sequence<Map2::dim>{});
}

template <typename Map1, typename Map2>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, ProductOf2Maps<Map1, Map2>::dim,
         Frame::NoFrame>
ProductOf2Maps<Map1, Map2>::jacobian(
    const std::array<T, dim>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  using UnwrappedT = tt::remove_cvref_wrap_t<T>;
  return product_detail::apply_jac(
      source_coords, map1_, map2_,
      [&time, &functions_of_time](const auto& point, const auto& map) noexcept {
        return CoordinateMap_detail::apply_jacobian(
            map, point, time, functions_of_time,
            domain::is_jacobian_time_dependent_t<
                std::decay_t<decltype(map)>,
                std::reference_wrapper<const UnwrappedT>>{});
      },
      std::make_index_sequence<Map1::dim>{},
      std::make_index_sequence<Map2::dim>{});
}

template <typename Map1, typename Map2>
void ProductOf2Maps<Map1, Map2>::pup(PUP::er& p) {
  p | map1_;
  p | map2_;
}

template <typename Map1, typename Map2>
bool operator!=(const ProductOf2Maps<Map1, Map2>& lhs,
                const ProductOf2Maps<Map1, Map2>& rhs) noexcept {
  return not(lhs == rhs);
}

template <typename Map1, typename Map2, typename Map3>
ProductOf3Maps<Map1, Map2, Map3>::ProductOf3Maps(Map1 map1, Map2 map2,
                                                 Map3 map3) noexcept
    : map1_(std::move(map1)), map2_(std::move(map2)), map3_(std::move(map3)) {}

template <typename Map1, typename Map2, typename Map3>
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, ProductOf3Maps<Map1, Map2, Map3>::dim>
ProductOf3Maps<Map1, Map2, Map3>::operator()(
    const std::array<T, dim>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  using UnwrappedT = tt::remove_cvref_wrap_t<T>;
  return {
      {CoordinateMap_detail::apply_map(
           map1_,
           std::array<std::reference_wrapper<const UnwrappedT>, 1>{
               {source_coords[0]}},
           time, functions_of_time, domain::is_map_time_dependent_t<Map1>{})[0],
       CoordinateMap_detail::apply_map(
           map2_,
           std::array<std::reference_wrapper<const UnwrappedT>, 1>{
               {source_coords[1]}},
           time, functions_of_time, domain::is_map_time_dependent_t<Map2>{})[0],
       CoordinateMap_detail::apply_map(
           map3_,
           std::array<std::reference_wrapper<const UnwrappedT>, 1>{
               {source_coords[2]}},
           time, functions_of_time,
           domain::is_map_time_dependent_t<Map3>{})[0]}};
}

template <typename Map1, typename Map2, typename Map3>
boost::optional<std::array<double, ProductOf3Maps<Map1, Map2, Map3>::dim>>
ProductOf3Maps<Map1, Map2, Map3>::inverse(
    const std::array<double, dim>& target_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  const auto c1 = CoordinateMap_detail::apply_inverse_map(
      map1_, std::array<double, 1>{{target_coords[0]}}, time, functions_of_time,
      domain::is_map_time_dependent_t<Map1>{});
  const auto c2 = CoordinateMap_detail::apply_inverse_map(
      map2_, std::array<double, 1>{{target_coords[1]}}, time, functions_of_time,
      domain::is_map_time_dependent_t<Map2>{});
  const auto c3 = CoordinateMap_detail::apply_inverse_map(
      map3_, std::array<double, 1>{{target_coords[2]}}, time, functions_of_time,
      domain::is_map_time_dependent_t<Map3>{});
  if (c1 and c2 and c3) {
    return {{{c1.get()[0], c2.get()[0], c3.get()[0]}}};
  } else {
    return boost::none;
  }
}

template <typename Map1, typename Map2, typename Map3>
template <typename T>
auto ProductOf3Maps<Map1, Map2, Map3>::frame_velocity(
    const std::array<T, dim>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept
    -> std::array<tt::remove_cvref_wrap_t<T>, dim> {
  using UnwrappedT = tt::remove_cvref_wrap_t<T>;
  return {
      {CoordinateMap_detail::apply_frame_velocity(
           map1_,
           std::array<std::reference_wrapper<const UnwrappedT>, 1>{
               {source_coords[0]}},
           time, functions_of_time, domain::is_map_time_dependent_t<Map1>{})[0],
       CoordinateMap_detail::apply_frame_velocity(
           map2_,
           std::array<std::reference_wrapper<const UnwrappedT>, 1>{
               {source_coords[1]}},
           time, functions_of_time, domain::is_map_time_dependent_t<Map2>{})[0],
       CoordinateMap_detail::apply_frame_velocity(
           map3_,
           std::array<std::reference_wrapper<const UnwrappedT>, 1>{
               {source_coords[2]}},
           time, functions_of_time,
           domain::is_map_time_dependent_t<Map3>{})[0]}};
}

template <typename Map1, typename Map2, typename Map3>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, ProductOf3Maps<Map1, Map2, Map3>::dim,
         Frame::NoFrame>
ProductOf3Maps<Map1, Map2, Map3>::inv_jacobian(
    const std::array<T, dim>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  using UnwrappedT = tt::remove_cvref_wrap_t<T>;
  tnsr::Ij<UnwrappedT, dim, Frame::NoFrame> inv_jacobian_matrix{
      make_with_value<UnwrappedT>(dereference_wrapper(source_coords[0]), 0.0)};
  get<0, 0>(inv_jacobian_matrix) =
      get<0, 0>(CoordinateMap_detail::apply_inverse_jacobian(
          map1_,
          std::array<std::reference_wrapper<const UnwrappedT>, 1>{
              {source_coords[0]}},
          time, functions_of_time,
          domain::is_jacobian_time_dependent_t<
              Map1, std::reference_wrapper<const UnwrappedT>>{}));
  get<1, 1>(inv_jacobian_matrix) =
      get<0, 0>(CoordinateMap_detail::apply_inverse_jacobian(
          map2_,
          std::array<std::reference_wrapper<const UnwrappedT>, 1>{
              {source_coords[1]}},
          time, functions_of_time,
          domain::is_jacobian_time_dependent_t<
              Map2, std::reference_wrapper<const UnwrappedT>>{}));
  get<2, 2>(inv_jacobian_matrix) =
      get<0, 0>(CoordinateMap_detail::apply_inverse_jacobian(
          map3_,
          std::array<std::reference_wrapper<const UnwrappedT>, 1>{
              {source_coords[2]}},
          time, functions_of_time,
          domain::is_jacobian_time_dependent_t<
              Map3, std::reference_wrapper<const UnwrappedT>>{}));
  return inv_jacobian_matrix;
}

template <typename Map1, typename Map2, typename Map3>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, ProductOf3Maps<Map1, Map2, Map3>::dim,
         Frame::NoFrame>
ProductOf3Maps<Map1, Map2, Map3>::jacobian(
    const std::array<T, dim>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  using UnwrappedT = tt::remove_cvref_wrap_t<T>;
  tnsr::Ij<UnwrappedT, dim, Frame::NoFrame> jacobian_matrix{
      make_with_value<UnwrappedT>(dereference_wrapper(source_coords[0]), 0.0)};
  get<0, 0>(jacobian_matrix) = get<0, 0>(CoordinateMap_detail::apply_jacobian(
      map1_,
      std::array<std::reference_wrapper<const UnwrappedT>, 1>{
          {source_coords[0]}},
      time, functions_of_time,
      domain::is_jacobian_time_dependent_t<
          Map1, std::reference_wrapper<const UnwrappedT>>{}));
  get<1, 1>(jacobian_matrix) = get<0, 0>(CoordinateMap_detail::apply_jacobian(
      map2_,
      std::array<std::reference_wrapper<const UnwrappedT>, 1>{
          {source_coords[1]}},
      time, functions_of_time,
      domain::is_jacobian_time_dependent_t<
          Map2, std::reference_wrapper<const UnwrappedT>>{}

      ));
  get<2, 2>(jacobian_matrix) = get<0, 0>(CoordinateMap_detail::apply_jacobian(
      map3_,
      std::array<std::reference_wrapper<const UnwrappedT>, 1>{
          {source_coords[2]}},
      time, functions_of_time,
      domain::is_jacobian_time_dependent_t<
          Map3, std::reference_wrapper<const UnwrappedT>>{}));
  return jacobian_matrix;
}
template <typename Map1, typename Map2, typename Map3>
void ProductOf3Maps<Map1, Map2, Map3>::pup(PUP::er& p) noexcept {
  p | map1_;
  p | map2_;
  p | map3_;
}

template <typename Map1, typename Map2, typename Map3>
bool operator!=(const ProductOf3Maps<Map1, Map2, Map3>& lhs,
                const ProductOf3Maps<Map1, Map2, Map3>& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace TimeDependent
}  // namespace CoordinateMaps
}  // namespace domain
