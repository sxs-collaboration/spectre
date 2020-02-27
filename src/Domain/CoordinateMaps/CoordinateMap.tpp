// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/CoordinateMaps/CoordinateMap.hpp"

#include <algorithm>
#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <pup.h>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/Identity.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMapHelpers.hpp"
#include "Domain/CoordinateMaps/TimeDependentHelpers.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/Tuple.hpp"

/// \cond
namespace domain {
template <typename SourceFrame, typename TargetFrame, typename... Maps>
CoordinateMap<SourceFrame, TargetFrame, Maps...>::CoordinateMap(Maps... maps)
    : maps_(std::move(maps)...) {}

namespace CoordinateMap_detail {
template <typename... Maps, size_t... Is>
bool is_identity_impl(const std::tuple<Maps...>& maps,
                      std::index_sequence<Is...> /*meta*/) noexcept {
  bool is_identity = true;
  const auto check_map_is_identity = [&is_identity,
                                      &maps](auto index) noexcept {
    if (is_identity) {
      is_identity = std::get<decltype(index)::value>(maps).is_identity();
    }
    return '0';
  };
  EXPAND_PACK_LEFT_TO_RIGHT(
      check_map_is_identity(std::integral_constant<size_t, Is>{}));
  return is_identity;
}
}  // namespace CoordinateMap_detail

template <typename SourceFrame, typename TargetFrame, typename... Maps>
bool CoordinateMap<SourceFrame, TargetFrame, Maps...>::is_identity() const
    noexcept {
  return CoordinateMap_detail::is_identity_impl(
      maps_, std::make_index_sequence<sizeof...(Maps)>{});
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
bool CoordinateMap<SourceFrame, TargetFrame,
                   Maps...>::inv_jacobian_is_time_dependent() const noexcept {
  return tmpl2::flat_any_v<
      domain::is_jacobian_time_dependent_t<Maps, double>::value...>;
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
bool CoordinateMap<SourceFrame, TargetFrame,
                   Maps...>::jacobian_is_time_dependent() const noexcept {
  return inv_jacobian_is_time_dependent();
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
template <typename T, size_t... Is>
tnsr::I<T, CoordinateMap<SourceFrame, TargetFrame, Maps...>::dim, TargetFrame>
CoordinateMap<SourceFrame, TargetFrame, Maps...>::call_impl(
    tnsr::I<T, dim, SourceFrame>&& source_point, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    std::index_sequence<Is...> /*meta*/) const noexcept {
  std::array<T, dim> mapped_point = make_array<T, dim>(std::move(source_point));

  EXPAND_PACK_LEFT_TO_RIGHT(make_overloader(
      [](const auto& the_map, std::array<T, dim>& point, const double /*t*/,
         const std::unordered_map<
             std::string,
             std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
         /*funcs_of_time*/,
         const std::false_type /*is_time_independent*/) noexcept {
        if (LIKELY(not the_map.is_identity())) {
          point = the_map(point);
        }
      },
      [](const auto& the_map, std::array<T, dim>& point, const double t,
         const std::unordered_map<
             std::string,
             std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
             funcs_of_time,
         const std::true_type /*is_time_dependent*/) noexcept {
        point = the_map(point, t, funcs_of_time);
      })(std::get<Is>(maps_), mapped_point, time, functions_of_time,
         domain::is_map_time_dependent_t<Maps>{}));

  return tnsr::I<T, dim, TargetFrame>(std::move(mapped_point));
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
template <typename T, size_t... Is>
boost::optional<tnsr::I<
    T, CoordinateMap<SourceFrame, TargetFrame, Maps...>::dim, SourceFrame>>
CoordinateMap<SourceFrame, TargetFrame, Maps...>::inverse_impl(
    tnsr::I<T, dim, TargetFrame>&& target_point, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    std::index_sequence<Is...> /*meta*/) const noexcept {
  boost::optional<std::array<T, dim>> mapped_point(
      make_array<T, dim>(std::move(target_point)));

  EXPAND_PACK_LEFT_TO_RIGHT(make_overloader(
      [](const auto& the_map, boost::optional<std::array<T, dim>>& point,
         const double /*t*/,
         const std::unordered_map<
             std::string,
             std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
         /*funcs_of_time*/,
         const std::false_type /*is_time_independent*/) noexcept {
        if (point) {
          if (LIKELY(not the_map.is_identity())) {
            point = the_map.inverse(point.get());
          }
        }
      },
      [](const auto& the_map, boost::optional<std::array<T, dim>>& point,
         const double t,
         const std::unordered_map<
             std::string,
             std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
             funcs_of_time,
         const std::true_type /*is_time_dependent*/) noexcept {
        if (point) {
          point = the_map.inverse(point.get(), t, funcs_of_time);
        }
        // this is the inverse function, so the iterator sequence below is
        // reversed
      })(std::get<sizeof...(Maps) - 1 - Is>(maps_), mapped_point, time,
         functions_of_time,
         domain::is_map_time_dependent_t<decltype(
             std::get<sizeof...(Maps) - 1 - Is>(maps_))>{}));

  return mapped_point
             ? tnsr::I<T, dim, SourceFrame>(std::move(mapped_point.get()))
             : boost::optional<tnsr::I<T, dim, SourceFrame>>{};
}

namespace detail {
template <typename T, typename Map, size_t Dim>
void get_jacobian(
    const gsl::not_null<tnsr::Ij<T, Dim, Frame::NoFrame>*> no_frame_jac,
    const Map& the_map, const std::array<T, Dim>& point, const double /*t*/,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
    /*funcs_of_time*/,
    std::false_type /*jacobian_is_time_dependent*/) noexcept {
  if (LIKELY(not the_map.is_identity())) {
    *no_frame_jac = the_map.jacobian(point);
  } else {
    *no_frame_jac = identity<Dim>(point[0]);
  }
}

template <typename T, typename Map, size_t Dim>
void get_jacobian(
    const gsl::not_null<tnsr::Ij<T, Dim, Frame::NoFrame>*> no_frame_jac,
    const Map& the_map, const std::array<T, Dim>& point, const double t,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        funcs_of_time,
    std::true_type /*jacobian_is_time_dependent*/) noexcept {
  *no_frame_jac = the_map.jacobian(point, t, funcs_of_time);
}

template <typename T, typename Map, size_t Dim>
void get_inv_jacobian(
    const gsl::not_null<tnsr::Ij<T, Dim, Frame::NoFrame>*> no_frame_inv_jac,
    const Map& the_map, const std::array<T, Dim>& point, const double /*t*/,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
    /*funcs_of_time*/,
    std::false_type /*jacobian_is_time_dependent*/) noexcept {
  if (LIKELY(not the_map.is_identity())) {
    *no_frame_inv_jac = the_map.inv_jacobian(point);
  } else {
    *no_frame_inv_jac = identity<Dim>(point[0]);
  }
}

template <typename T, typename Map, size_t Dim>
void get_inv_jacobian(
    const gsl::not_null<tnsr::Ij<T, Dim, Frame::NoFrame>*> no_frame_inv_jac,
    const Map& the_map, const std::array<T, Dim>& point, const double t,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        funcs_of_time,
    std::true_type /*jacobian_is_time_dependent*/) noexcept {
  *no_frame_inv_jac = the_map.inv_jacobian(point, t, funcs_of_time);
}

template <typename T, size_t Dim, typename SourceFrame, typename TargetFrame>
void multiply_jacobian(
    const gsl::not_null<Jacobian<T, Dim, SourceFrame, TargetFrame>*> jac,
    const tnsr::Ij<T, Dim, Frame::NoFrame>& noframe_jac) noexcept {
  std::array<T, Dim> temp{};
  for (size_t source = 0; source < Dim; ++source) {
    for (size_t target = 0; target < Dim; ++target) {
      gsl::at(temp, target) = noframe_jac.get(target, 0) * jac->get(0, source);
      for (size_t dummy = 1; dummy < Dim; ++dummy) {
        gsl::at(temp, target) +=
            noframe_jac.get(target, dummy) * jac->get(dummy, source);
      }
    }
    for (size_t target = 0; target < Dim; ++target) {
      jac->get(target, source) = std::move(gsl::at(temp, target));
    }
  }
}

template <typename T, size_t Dim, typename SourceFrame, typename TargetFrame>
void multiply_inv_jacobian(
    const gsl::not_null<Jacobian<T, Dim, SourceFrame, TargetFrame>*> inv_jac,
    const tnsr::Ij<T, Dim, Frame::NoFrame>& noframe_inv_jac) noexcept {
  std::array<T, Dim> temp{};
  for (size_t source = 0; source < Dim; ++source) {
    for (size_t target = 0; target < Dim; ++target) {
      gsl::at(temp, target) =
          inv_jac->get(source, 0) * noframe_inv_jac.get(0, target);
      for (size_t dummy = 1; dummy < Dim; ++dummy) {
        gsl::at(temp, target) +=
            inv_jac->get(source, dummy) * noframe_inv_jac.get(dummy, target);
      }
    }
    for (size_t target = 0; target < Dim; ++target) {
      inv_jac->get(source, target) = std::move(gsl::at(temp, target));
    }
  }
}
}  // namespace detail

template <typename SourceFrame, typename TargetFrame, typename... Maps>
template <typename T>
auto CoordinateMap<SourceFrame, TargetFrame, Maps...>::inv_jacobian_impl(
    tnsr::I<T, dim, SourceFrame>&& source_point, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept
    -> InverseJacobian<T, dim, SourceFrame, TargetFrame> {
  std::array<T, dim> mapped_point = make_array<T, dim>(std::move(source_point));

  InverseJacobian<T, dim, SourceFrame, TargetFrame> inv_jac{};

  tuple_transform(
      maps_, [&inv_jac, &mapped_point, time, &functions_of_time](
                 const auto& map, auto index) noexcept {
        constexpr size_t count = decltype(index)::value;
        using Map = std::decay_t<decltype(map)>;

        tnsr::Ij<T, dim, Frame::NoFrame> noframe_inv_jac{};

        if (UNLIKELY(count == 0)) {
          detail::get_inv_jacobian(
              make_not_null(&noframe_inv_jac), map, mapped_point, time,
              functions_of_time,
              domain::is_jacobian_time_dependent_t<Map, T>{});
          for (size_t source = 0; source < dim; ++source) {
            for (size_t target = 0; target < dim; ++target) {
              inv_jac.get(source, target) =
                  std::move(noframe_inv_jac.get(source, target));
            }
          }
        } else if (LIKELY(not map.is_identity())) {
          detail::get_inv_jacobian(
              make_not_null(&noframe_inv_jac), map, mapped_point, time,
              functions_of_time,
              domain::is_jacobian_time_dependent_t<Map, T>{});
          detail::multiply_inv_jacobian(make_not_null(&inv_jac),
                                        noframe_inv_jac);
        }

        // Compute the source coordinates for the next map, only if we are not
        // the last map and the map is not the identity.
        if (not map.is_identity() and count + 1 != sizeof...(Maps)) {
          CoordinateMap_detail::apply_map(
              make_not_null(&mapped_point), map, time, functions_of_time,
              domain::is_map_time_dependent_t<decltype(map)>{});
        }
      });
  return inv_jac;
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
template <typename T>
auto CoordinateMap<SourceFrame, TargetFrame, Maps...>::jacobian_impl(
    tnsr::I<T, dim, SourceFrame>&& source_point, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept
    -> Jacobian<T, dim, SourceFrame, TargetFrame> {
  std::array<T, dim> mapped_point = make_array<T, dim>(std::move(source_point));
  Jacobian<T, dim, SourceFrame, TargetFrame> jac{};

  tuple_transform(
      maps_, [&jac, &mapped_point, time, &functions_of_time](
                 const auto& map, auto index) noexcept {
        constexpr size_t count = decltype(index)::value;
        using Map = std::decay_t<decltype(map)>;

        tnsr::Ij<T, dim, Frame::NoFrame> noframe_jac{};

        if (UNLIKELY(count == 0)) {
          detail::get_jacobian(make_not_null(&noframe_jac), map, mapped_point,
                               time, functions_of_time,
                               domain::is_jacobian_time_dependent_t<Map, T>{});
          for (size_t target = 0; target < dim; ++target) {
            for (size_t source = 0; source < dim; ++source) {
              jac.get(target, source) =
                  std::move(noframe_jac.get(target, source));
            }
          }
        } else if (LIKELY(not map.is_identity())) {
          detail::get_jacobian(make_not_null(&noframe_jac), map, mapped_point,
                               time, functions_of_time,
                               domain::is_jacobian_time_dependent_t<Map, T>{});
          detail::multiply_jacobian(make_not_null(&jac), noframe_jac);
        }

        // Compute the source coordinates for the next map, only if we are not
        // the last map and the map is not the identity.
        if (not map.is_identity() and count + 1 != sizeof...(Maps)) {
          CoordinateMap_detail::apply_map(
              make_not_null(&mapped_point), map, time, functions_of_time,
              domain::is_map_time_dependent_t<decltype(map)>{});
        }
      });
  return jac;
}

namespace detail {
template <typename T, typename Map, size_t Dim,
          Requires<not domain::is_map_time_dependent_v<Map>> = nullptr>
std::array<T, Dim> get_frame_velocity(
    const Map& /*the_map*/, const std::array<T, Dim>& /*point*/,
    const double /*t*/,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
    /*funcs_of_time*/) noexcept {
  return std::array<T, Dim>{};
}

template <typename T, typename Map, size_t Dim,
          Requires<domain::is_map_time_dependent_v<Map>> = nullptr>
std::array<T, Dim> get_frame_velocity(
    const Map& the_map, const std::array<T, Dim>& point, const double t,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        funcs_of_time) noexcept {
  return the_map.frame_velocity(point, t, funcs_of_time);
}
}  // namespace detail

template <typename SourceFrame, typename TargetFrame, typename... Maps>
template <typename T>
auto CoordinateMap<SourceFrame, TargetFrame, Maps...>::
    coords_frame_velocity_jacobians_impl(
        tnsr::I<T, dim, SourceFrame> source_point, const double time,
        const std::unordered_map<
            std::string,
            std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
            functions_of_time) const noexcept
    -> std::tuple<tnsr::I<T, dim, TargetFrame>,
                  InverseJacobian<T, dim, SourceFrame, TargetFrame>,
                  Jacobian<T, dim, SourceFrame, TargetFrame>,
                  tnsr::I<T, dim, TargetFrame>> {
  std::array<T, dim> mapped_point = make_array<T, dim>(std::move(source_point));
  InverseJacobian<T, dim, SourceFrame, TargetFrame> inv_jac{};
  Jacobian<T, dim, SourceFrame, TargetFrame> jac{};
  tnsr::I<T, dim, TargetFrame> frame_velocity{};

  tuple_transform(
      maps_,
      [&frame_velocity, &inv_jac, &jac, &mapped_point, time,
       &functions_of_time](const auto& map, auto index,
                           const std::tuple<Maps...>& maps) noexcept {
        constexpr size_t count = decltype(index)::value;
        using Map = std::decay_t<decltype(map)>;

        tnsr::Ij<T, dim, Frame::NoFrame> noframe_jac{};
        tnsr::Ij<T, dim, Frame::NoFrame> noframe_inv_jac{};

        if (UNLIKELY(count == 0)) {
          // Set Jacobian and inverse Jacobian
          detail::get_inv_jacobian(
              make_not_null(&noframe_inv_jac), map, mapped_point, time,
              functions_of_time,
              domain::is_jacobian_time_dependent_t<Map, T>{});
          detail::get_jacobian(make_not_null(&noframe_jac), map, mapped_point,
                               time, functions_of_time,
                               domain::is_jacobian_time_dependent_t<Map, T>{});
          for (size_t target = 0; target < dim; ++target) {
            for (size_t source = 0; source < dim; ++source) {
              jac.get(target, source) =
                  std::move(noframe_jac.get(target, source));
              inv_jac.get(source, target) =
                  std::move(noframe_inv_jac.get(source, target));
            }
          }

          // Set frame velocity
          if (domain::is_map_time_dependent_v<
                  std::tuple_element_t<0, std::decay_t<decltype(maps)>>>) {
            std::array<T, dim> noframe_frame_velocity =
                detail::get_frame_velocity(map, mapped_point, time,
                                           functions_of_time);
            for (size_t i = 0; i < dim; ++i) {
              frame_velocity.get(i) =
                  std::move(gsl::at(noframe_frame_velocity, i));
            }
          } else {
            // If the first map is time-independent the velocity is initialized
            // to zero
            for (size_t i = 0; i < frame_velocity.size(); ++i) {
              frame_velocity[i] = make_with_value<T>(get<0, 0>(jac), 0.0);
            }
          }
        } else if (LIKELY(not map.is_identity())) {
          // WARNING: we have assumed that if the map is the identity the frame
          // velocity is also zero. That is, we do not optimize for the map
          // being instantaneously zero.

          detail::get_inv_jacobian(
              make_not_null(&noframe_inv_jac), map, mapped_point, time,
              functions_of_time,
              domain::is_jacobian_time_dependent_t<Map, T>{});
          detail::get_jacobian(make_not_null(&noframe_jac), map, mapped_point,
                               time, functions_of_time,
                               domain::is_jacobian_time_dependent_t<Map, T>{});

          // Perform matrix multiplication for Jacobian and inverse Jacobian
          detail::multiply_inv_jacobian(make_not_null(&inv_jac),
                                        noframe_inv_jac);
          detail::multiply_jacobian(make_not_null(&jac), noframe_jac);

          // Set frame velocity, only if map is time-dependent
          std::array<T, dim> noframe_frame_velocity{};
          if (domain::is_map_time_dependent_v<
                  std::tuple_element_t<count, std::decay_t<decltype(maps)>>>) {
            noframe_frame_velocity = detail::get_frame_velocity(
                map, mapped_point, time, functions_of_time);
            for (size_t target_frame_index = 0; target_frame_index < dim;
                 ++target_frame_index) {
              for (size_t source_frame_index = 0; source_frame_index < dim;
                   ++source_frame_index) {
                gsl::at(noframe_frame_velocity, target_frame_index) +=
                    noframe_jac.get(target_frame_index, source_frame_index) *
                    frame_velocity.get(source_frame_index);
              }
            }
          } else {
            for (size_t target_frame_index = 0; target_frame_index < dim;
                 ++target_frame_index) {
              size_t source_frame_index = 0;
              gsl::at(noframe_frame_velocity, target_frame_index) =
                  noframe_jac.get(target_frame_index, source_frame_index) *
                  frame_velocity.get(source_frame_index);
              for (source_frame_index = 1; source_frame_index < dim;
                   ++source_frame_index) {
                gsl::at(noframe_frame_velocity, target_frame_index) +=
                    noframe_jac.get(target_frame_index, source_frame_index) *
                    frame_velocity.get(source_frame_index);
              }
            }
          }
          for (size_t target_frame_index = 0; target_frame_index < dim;
               ++target_frame_index) {
            using std::swap;
            swap(gsl::at(noframe_frame_velocity, target_frame_index),
                 frame_velocity.get(target_frame_index));
          }
        }

        // Update to the next mapped point
        CoordinateMap_detail::apply_map(
            make_not_null(&mapped_point), map, time, functions_of_time,
            domain::is_map_time_dependent_t<decltype(map)>{});
      },
      maps_);
  return std::tuple<tnsr::I<T, dim, TargetFrame>,
                    InverseJacobian<T, dim, SourceFrame, TargetFrame>,
                    Jacobian<T, dim, SourceFrame, TargetFrame>,
                    tnsr::I<T, dim, TargetFrame>>{
      tnsr::I<T, dim, TargetFrame>(std::move(mapped_point)), std::move(inv_jac),
      std::move(jac), std::move(frame_velocity)};
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
template <size_t... Is>
auto CoordinateMap<SourceFrame, TargetFrame, Maps...>::get_to_grid_frame_impl(
    std::index_sequence<Is...> /*meta*/) const noexcept
    -> std::unique_ptr<CoordinateMapBase<SourceFrame, Frame::Grid, dim>> {
  return std::make_unique<CoordinateMap<SourceFrame, Frame::Grid, Maps...>>(
      std::get<Is>(maps_)...);
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
bool operator!=(
    const CoordinateMap<SourceFrame, TargetFrame, Maps...>& lhs,
    const CoordinateMap<SourceFrame, TargetFrame, Maps...>& rhs) noexcept {
  return not(lhs == rhs);
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
auto make_coordinate_map(Maps&&... maps) noexcept
    -> CoordinateMap<SourceFrame, TargetFrame, std::decay_t<Maps>...> {
  return CoordinateMap<SourceFrame, TargetFrame, std::decay_t<Maps>...>(
      std::forward<Maps>(maps)...);
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
auto make_coordinate_map_base(Maps&&... maps) noexcept
    -> std::unique_ptr<CoordinateMapBase<
        SourceFrame, TargetFrame,
        CoordinateMap<SourceFrame, TargetFrame, std::decay_t<Maps>...>::dim>> {
  return std::make_unique<
      CoordinateMap<SourceFrame, TargetFrame, std::decay_t<Maps>...>>(
      std::forward<Maps>(maps)...);
}

template <typename SourceFrame, typename TargetFrame, typename Arg0,
          typename... Args>
auto make_vector_coordinate_map_base(Arg0&& arg_0,
                                     Args&&... remaining_args) noexcept
    -> std::vector<std::unique_ptr<
        CoordinateMapBase<SourceFrame, TargetFrame, std::decay_t<Arg0>::dim>>> {
  std::vector<std::unique_ptr<
      CoordinateMapBase<SourceFrame, TargetFrame, std::decay_t<Arg0>::dim>>>
      return_vector;
  return_vector.reserve(sizeof...(Args) + 1);
  return_vector.emplace_back(make_coordinate_map_base<SourceFrame, TargetFrame>(
      std::forward<Arg0>(arg_0)));
  EXPAND_PACK_LEFT_TO_RIGHT(return_vector.emplace_back(
      make_coordinate_map_base<SourceFrame, TargetFrame>(
          std::forward<Args>(remaining_args))));
  return return_vector;
}

template <typename SourceFrame, typename TargetFrame, size_t Dim, typename Map,
          typename... Maps>
auto make_vector_coordinate_map_base(std::vector<Map> maps,
                                     const Maps&... remaining_maps) noexcept
    -> std::vector<
        std::unique_ptr<CoordinateMapBase<SourceFrame, TargetFrame, Dim>>> {
  std::vector<std::unique_ptr<CoordinateMapBase<SourceFrame, TargetFrame, Dim>>>
      return_vector;
  return_vector.reserve(sizeof...(Maps) + 1);
  for (auto& map : maps) {
    return_vector.emplace_back(
        make_coordinate_map_base<SourceFrame, TargetFrame>(std::move(map),
                                                           remaining_maps...));
  }
  return return_vector;
}

template <typename NewMap, typename SourceFrame, typename TargetFrame,
          typename... Maps, size_t... Is>
CoordinateMap<SourceFrame, TargetFrame, Maps..., NewMap> push_back_impl(
    CoordinateMap<SourceFrame, TargetFrame, Maps...>&& old_map, NewMap new_map,
    std::index_sequence<Is...> /*meta*/) noexcept {
  return CoordinateMap<SourceFrame, TargetFrame, Maps..., NewMap>{
      std::move(std::get<Is>(old_map.maps_))..., std::move(new_map)};
}

template <typename... NewMaps, typename SourceFrame, typename TargetFrame,
          typename... Maps, size_t... Is, size_t... Js>
CoordinateMap<SourceFrame, TargetFrame, Maps..., NewMaps...> push_back_impl(
    CoordinateMap<SourceFrame, TargetFrame, Maps...>&& old_map,
    CoordinateMap<SourceFrame, TargetFrame, NewMaps...> new_map,
    std::index_sequence<Is...> /*meta*/,
    std::index_sequence<Js...> /*meta*/) noexcept {
  return CoordinateMap<SourceFrame, TargetFrame, Maps..., NewMaps...>{
      std::move(std::get<Is>(old_map.maps_))...,
      std::move(std::get<Js>(new_map.maps_))...};
}

template <typename NewMap, typename SourceFrame, typename TargetFrame,
          typename... Maps, size_t... Is>
CoordinateMap<SourceFrame, TargetFrame, NewMap, Maps...> push_front_impl(
    CoordinateMap<SourceFrame, TargetFrame, Maps...>&& old_map, NewMap new_map,
    std::index_sequence<Is...> /*meta*/) noexcept {
  return CoordinateMap<SourceFrame, TargetFrame, Maps..., NewMap>{
      std::move(new_map), std::move(std::get<Is>(old_map.maps_))...};
}

template <typename SourceFrame, typename TargetFrame, typename... Maps,
          typename NewMap>
CoordinateMap<SourceFrame, TargetFrame, Maps..., NewMap> push_back(
    CoordinateMap<SourceFrame, TargetFrame, Maps...> old_map,
    NewMap new_map) noexcept {
  return push_back_impl(std::move(old_map), std::move(new_map),
                        std::make_index_sequence<sizeof...(Maps)>{});
}

template <typename SourceFrame, typename TargetFrame, typename... Maps,
          typename... NewMaps>
CoordinateMap<SourceFrame, TargetFrame, Maps..., NewMaps...> push_back(
    CoordinateMap<SourceFrame, TargetFrame, Maps...> old_map,
    CoordinateMap<SourceFrame, TargetFrame, NewMaps...> new_map) noexcept {
  return push_back_impl(std::move(old_map), std::move(new_map),
                        std::make_index_sequence<sizeof...(Maps)>{},
                        std::make_index_sequence<sizeof...(NewMaps)>{});
}

template <typename SourceFrame, typename TargetFrame, typename... Maps,
          typename NewMap>
CoordinateMap<SourceFrame, TargetFrame, NewMap, Maps...> push_front(
    CoordinateMap<SourceFrame, TargetFrame, Maps...> old_map,
    NewMap new_map) noexcept {
  return push_front_impl(std::move(old_map), std::move(new_map),
                         std::make_index_sequence<sizeof...(Maps)>{});
}

template <typename SourceFrame, typename TargetFrame, typename... Maps,
          typename... NewMaps>
CoordinateMap<SourceFrame, TargetFrame, NewMaps..., Maps...> push_front(
    CoordinateMap<SourceFrame, TargetFrame, Maps...> old_map,
    CoordinateMap<SourceFrame, TargetFrame, NewMaps...> new_map) noexcept {
  return push_back_impl(std::move(new_map), std::move(old_map),
                        std::make_index_sequence<sizeof...(NewMaps)>{},
                        std::make_index_sequence<sizeof...(Maps)>{});
}
}  // namespace domain
/// \endcond
