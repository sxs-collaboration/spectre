// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/CoordinateMaps/CoordinateMap.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Identity.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMapHelpers.hpp"
#include "Domain/CoordinateMaps/TimeDependentHelpers.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/Tuple.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
namespace domain {
namespace CoordinateMap_detail {
CREATE_IS_CALLABLE(function_of_time_names)
CREATE_IS_CALLABLE_V(function_of_time_names)

template <size_t Dim, typename T>
using combined_coords_frame_velocity_jacs_t =
    decltype(std::declval<T>().coords_frame_velocity_jacobian(
        std::declval<gsl::not_null<std::array<DataVector, Dim>*>>(),
        std::declval<gsl::not_null<std::array<DataVector, Dim>*>>(),
        std::declval<
            gsl::not_null<tnsr::Ij<DataVector, Dim, Frame::NoFrame>*>>(),
        std::declval<const double>(),
        std::declval<const std::unordered_map<
            std::string,
            std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&>()));

template <size_t Dim, typename T, typename = std::void_t<>>
struct has_combined_coords_frame_velocity_jacs : std::false_type {};

template <size_t Dim, typename T>
struct has_combined_coords_frame_velocity_jacs<
    Dim, T, std::void_t<combined_coords_frame_velocity_jacs_t<Dim, T>>>
    : std::true_type {};

template <size_t Dim, typename T>
inline constexpr bool has_combined_coords_frame_velocity_jacs_v =
    has_combined_coords_frame_velocity_jacs<Dim, T>::value;

template <typename T>
struct map_type {
  using type = T;
};

template <typename T>
struct map_type<std::unique_ptr<T>> {
  using type = T;
};

template <typename T>
using map_type_t = typename map_type<T>::type;

template <typename... Maps, size_t... Is>
std::unordered_set<std::string> initialize_names(
    const std::tuple<Maps...>& maps, std::index_sequence<Is...> /*meta*/) {
  std::unordered_set<std::string> function_of_time_names{};
  const auto add_names = [&function_of_time_names, &maps](auto index) {
    const auto& map = std::get<decltype(index)::value>(maps);
    using TupleMap = std::decay_t<decltype(map)>;
    constexpr bool map_is_unique_ptr = tt::is_a_v<std::unique_ptr, TupleMap>;
    using Map = map_type_t<TupleMap>;
    if constexpr (is_function_of_time_names_callable_v<Map>) {
      if constexpr (map_is_unique_ptr) {
        const auto& names = map->function_of_time_names();
        function_of_time_names.insert(names.begin(), names.end());
      } else {
        const auto& names = map.function_of_time_names();
        function_of_time_names.insert(names.begin(), names.end());
      }
    } else {
      (void)function_of_time_names;
    }
  };
  EXPAND_PACK_LEFT_TO_RIGHT(add_names(std::integral_constant<size_t, Is>{}));

  return function_of_time_names;
}
}  // namespace CoordinateMap_detail

template <typename SourceFrame, typename TargetFrame, typename... Maps>
CoordinateMap<SourceFrame, TargetFrame, Maps...>::CoordinateMap(Maps... maps)
    : maps_(std::move(maps)...),
      function_of_time_names_(CoordinateMap_detail::initialize_names(
          maps_, std::make_index_sequence<sizeof...(Maps)>{})) {}

namespace CoordinateMap_detail {
template <typename... Maps, size_t... Is>
bool is_identity_impl(const std::tuple<Maps...>& maps,
                      std::index_sequence<Is...> /*meta*/) {
  bool is_identity = true;
  const auto check_map_is_identity = [&is_identity, &maps](auto index) {
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
bool CoordinateMap<SourceFrame, TargetFrame, Maps...>::is_identity() const {
  return CoordinateMap_detail::is_identity_impl(
      maps_, std::make_index_sequence<sizeof...(Maps)>{});
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
bool CoordinateMap<SourceFrame, TargetFrame,
                   Maps...>::inv_jacobian_is_time_dependent() const {
  return tmpl2::flat_any_v<
      domain::is_jacobian_time_dependent_t<Maps, double>::value...>;
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
bool CoordinateMap<SourceFrame, TargetFrame,
                   Maps...>::jacobian_is_time_dependent() const {
  return inv_jacobian_is_time_dependent();
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
void CoordinateMap<SourceFrame, TargetFrame, Maps...>::check_functions_of_time(
    [[maybe_unused]] const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
// Even though an assert is already in debug mode, we also want to avoid the
// loop
#ifdef SPECTRE_DEBUG
  for (const std::string& name : function_of_time_names_) {
    ASSERT(functions_of_time.count(name) == 1,
           "The function of time '" << name
                                    << "' is not one of the known functions of "
                                       "time. The known functions of time are: "
                                    << keys_of(functions_of_time));
  }
#endif
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
template <typename T, size_t... Is>
tnsr::I<T, CoordinateMap<SourceFrame, TargetFrame, Maps...>::dim, TargetFrame>
CoordinateMap<SourceFrame, TargetFrame, Maps...>::call_impl(
    tnsr::I<T, dim, SourceFrame>&& source_point, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    std::index_sequence<Is...> /*meta*/) const {
  check_functions_of_time(functions_of_time);
  std::array<T, dim> mapped_point = make_array<T, dim>(std::move(source_point));

  EXPAND_PACK_LEFT_TO_RIGHT(
      [](const auto& the_map, std::array<T, dim>& point, const double t,
         const std::unordered_map<
             std::string,
             std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
             funcs_of_time) {
        if constexpr (domain::is_map_time_dependent_t<decltype(the_map)>{}) {
          point = the_map(point, t, funcs_of_time);
        } else {
          (void)t;
          (void)funcs_of_time;
          if (LIKELY(not the_map.is_identity())) {
            point = the_map(point);
          }
        }
      }(std::get<Is>(maps_), mapped_point, time, functions_of_time));

  return tnsr::I<T, dim, TargetFrame>(std::move(mapped_point));
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
template <typename T, size_t... Is>
std::optional<tnsr::I<T, CoordinateMap<SourceFrame, TargetFrame, Maps...>::dim,
                      SourceFrame>>
CoordinateMap<SourceFrame, TargetFrame, Maps...>::inverse_impl(
    tnsr::I<T, dim, TargetFrame>&& target_point, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    std::index_sequence<Is...> /*meta*/) const {
  check_functions_of_time(functions_of_time);
  std::optional<std::array<T, dim>> mapped_point(
      make_array<T, dim>(std::move(target_point)));

  EXPAND_PACK_LEFT_TO_RIGHT(
      [](const auto& the_map, std::optional<std::array<T, dim>>& point,
         const double t,
         const std::unordered_map<
             std::string,
             std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
             funcs_of_time) {
        if constexpr (domain::is_map_time_dependent_t<decltype(the_map)>{}) {
          if (point.has_value()) {
            point = the_map.inverse(point.value(), t, funcs_of_time);
          }
        } else {
          (void)t;
          (void)funcs_of_time;
          if (point.has_value()) {
            if (LIKELY(not the_map.is_identity())) {
              point = the_map.inverse(point.value());
            }
          }
        }
        // this is the inverse function, so the iterator sequence below is
        // reversed
      }(std::get<sizeof...(Maps) - 1 - Is>(maps_), mapped_point, time,
        functions_of_time));

  return mapped_point
             ? tnsr::I<T, dim, SourceFrame>(std::move(mapped_point.value()))
             : std::optional<tnsr::I<T, dim, SourceFrame>>{};
}

namespace detail {
template <typename T, typename Map, size_t Dim>
void get_jacobian(
    const gsl::not_null<tnsr::Ij<T, Dim, Frame::NoFrame>*> no_frame_jac,
    const Map& the_map, const std::array<T, Dim>& point, const double /*t*/,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
    /*funcs_of_time*/,
    std::false_type /*jacobian_is_time_dependent*/) {
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
    std::true_type /*jacobian_is_time_dependent*/) {
  *no_frame_jac = the_map.jacobian(point, t, funcs_of_time);
}

template <typename T, typename Map, size_t Dim>
void get_inv_jacobian(
    const gsl::not_null<tnsr::Ij<T, Dim, Frame::NoFrame>*> no_frame_inv_jac,
    const Map& the_map, const std::array<T, Dim>& point, const double /*t*/,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
    /*funcs_of_time*/,
    std::false_type /*jacobian_is_time_dependent*/) {
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
    std::true_type /*jacobian_is_time_dependent*/) {
  *no_frame_inv_jac = the_map.inv_jacobian(point, t, funcs_of_time);
}

template <typename T, size_t Dim, typename SourceFrame, typename TargetFrame>
void multiply_jacobian(
    const gsl::not_null<Jacobian<T, Dim, SourceFrame, TargetFrame>*> jac,
    const tnsr::Ij<T, Dim, Frame::NoFrame>& noframe_jac) {
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
    const tnsr::Ij<T, Dim, Frame::NoFrame>& noframe_inv_jac) {
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
        functions_of_time) const
    -> InverseJacobian<T, dim, SourceFrame, TargetFrame> {
  check_functions_of_time(functions_of_time);
  std::array<T, dim> mapped_point = make_array<T, dim>(std::move(source_point));

  InverseJacobian<T, dim, SourceFrame, TargetFrame> inv_jac{};

  tuple_transform(maps_, [&inv_jac, &mapped_point, time, &functions_of_time](
                             const auto& map, auto index) {
    constexpr size_t count = decltype(index)::value;
    using Map = std::decay_t<decltype(map)>;

    tnsr::Ij<T, dim, Frame::NoFrame> noframe_inv_jac{};

    if (UNLIKELY(count == 0)) {
      ::domain::detail::get_inv_jacobian(
          make_not_null(&noframe_inv_jac), map, mapped_point, time,
          functions_of_time, domain::is_jacobian_time_dependent_t<Map, T>{});
      for (size_t source = 0; source < dim; ++source) {
        for (size_t target = 0; target < dim; ++target) {
          inv_jac.get(source, target) =
              std::move(noframe_inv_jac.get(source, target));
        }
      }
    } else if (LIKELY(not map.is_identity())) {
      ::domain::detail::get_inv_jacobian(
          make_not_null(&noframe_inv_jac), map, mapped_point, time,
          functions_of_time, domain::is_jacobian_time_dependent_t<Map, T>{});
      ::domain::detail::multiply_inv_jacobian(make_not_null(&inv_jac),
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
        functions_of_time) const -> Jacobian<T, dim, SourceFrame, TargetFrame> {
  check_functions_of_time(functions_of_time);
  std::array<T, dim> mapped_point = make_array<T, dim>(std::move(source_point));
  Jacobian<T, dim, SourceFrame, TargetFrame> jac{};

  tuple_transform(maps_, [&jac, &mapped_point, time, &functions_of_time](
                             const auto& map, auto index) {
    constexpr size_t count = decltype(index)::value;
    using Map = std::decay_t<decltype(map)>;

    tnsr::Ij<T, dim, Frame::NoFrame> noframe_jac{};

    if (UNLIKELY(count == 0)) {
      ::domain::detail::get_jacobian(
          make_not_null(&noframe_jac), map, mapped_point, time,
          functions_of_time, domain::is_jacobian_time_dependent_t<Map, T>{});
      for (size_t target = 0; target < dim; ++target) {
        for (size_t source = 0; source < dim; ++source) {
          jac.get(target, source) = std::move(noframe_jac.get(target, source));
        }
      }
    } else if (LIKELY(not map.is_identity())) {
      ::domain::detail::get_jacobian(
          make_not_null(&noframe_jac), map, mapped_point, time,
          functions_of_time, domain::is_jacobian_time_dependent_t<Map, T>{});
      ::domain::detail::multiply_jacobian(make_not_null(&jac), noframe_jac);
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
    /*funcs_of_time*/) {
  return std::array<T, Dim>{};
}

template <typename T, typename Map, size_t Dim,
          Requires<domain::is_map_time_dependent_v<Map>> = nullptr>
std::array<T, Dim> get_frame_velocity(
    const Map& the_map, const std::array<T, Dim>& point, const double t,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        funcs_of_time) {
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
            functions_of_time) const
    -> std::tuple<tnsr::I<T, dim, TargetFrame>,
                  InverseJacobian<T, dim, SourceFrame, TargetFrame>,
                  Jacobian<T, dim, SourceFrame, TargetFrame>,
                  tnsr::I<T, dim, TargetFrame>> {
  check_functions_of_time(functions_of_time);
  std::array<T, dim> mapped_point = make_array<T, dim>(std::move(source_point));
  Jacobian<T, dim, SourceFrame, TargetFrame> jac{};
  tnsr::I<T, dim, TargetFrame> frame_velocity{};

  tuple_transform(
      maps_,
      [&frame_velocity, &jac, &mapped_point, time, &functions_of_time](
          const auto& map, auto index, const std::tuple<Maps...>& /*maps*/) {
        constexpr size_t count = decltype(index)::value;
        using Map = std::decay_t<decltype(map)>;
        static constexpr bool is_time_dependent =
            domain::is_map_time_dependent_v<Map>;
        static constexpr bool use_combined_call =
            CoordinateMap_detail::has_combined_coords_frame_velocity_jacs_v<
                dim, Map> and
            std::is_same_v<T, DataVector>;

        [[maybe_unused]] std::array<T, dim> noframe_frame_velocity{};
        tnsr::Ij<T, dim, Frame::NoFrame> noframe_jac{};
        if constexpr (use_combined_call) {
          map.coords_frame_velocity_jacobian(
              make_not_null(&mapped_point),
              make_not_null(&noframe_frame_velocity),
              make_not_null(&noframe_jac), time, functions_of_time);
        } else {
          // if the map is the identity we do not compute it unless it is the
          // first map, then it is used for initialization
          if (not(map.is_identity() and count != 0)) {
            ::domain::detail::get_jacobian(
                make_not_null(&noframe_jac), map, mapped_point, time,
                functions_of_time,
                domain::is_jacobian_time_dependent_t<Map, T>{});
            if constexpr (is_time_dependent) {
              noframe_frame_velocity = domain::detail::get_frame_velocity(
                  map, mapped_point, time, functions_of_time);
            }
            CoordinateMap_detail::apply_map(
                make_not_null(&mapped_point), map, time, functions_of_time,
                domain::is_map_time_dependent_t<decltype(map)>{});
          }
        }
        if constexpr (count == 0) {
          for (size_t target = 0; target < dim; ++target) {
            for (size_t source = 0; source < dim; ++source) {
              jac.get(target, source) =
                  std::move(noframe_jac.get(target, source));
            }
          }
          // Set frame velocity
          if constexpr (is_time_dependent) {
            for (size_t i = 0; i < dim; ++i) {
              frame_velocity.get(i) =
                  std::move(gsl::at(noframe_frame_velocity, i));
            }
          } else {
            // If the first map is time-independent the velocity is
            // initialized to zero
            for (size_t i = 0; i < frame_velocity.size(); ++i) {
              frame_velocity[i] = make_with_value<T>(get<0, 0>(jac), 0.0);
            }
          }
        } else if (LIKELY(not map.is_identity())) {
          // WARNING: we have assumed that if the map is the identity the frame
          // velocity is also zero. That is, we do not optimize for the map
          // being instantaneously zero.
          ::domain::detail::multiply_jacobian(make_not_null(&jac), noframe_jac);

          if constexpr (is_time_dependent) {
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
      },
      maps_);
  return std::tuple<tnsr::I<T, dim, TargetFrame>,
                    InverseJacobian<T, dim, SourceFrame, TargetFrame>,
                    Jacobian<T, dim, SourceFrame, TargetFrame>,
                    tnsr::I<T, dim, TargetFrame>>{
      tnsr::I<T, dim, TargetFrame>(std::move(mapped_point)),
      determinant_and_inverse(jac).second, std::move(jac),
      std::move(frame_velocity)};
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
template <size_t... Is>
auto CoordinateMap<SourceFrame, TargetFrame, Maps...>::get_to_grid_frame_impl(
    std::index_sequence<Is...> /*meta*/) const
    -> std::unique_ptr<CoordinateMapBase<SourceFrame, Frame::Grid, dim>> {
  return std::make_unique<CoordinateMap<SourceFrame, Frame::Grid, Maps...>>(
      std::get<Is>(maps_)...);
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
bool operator!=(const CoordinateMap<SourceFrame, TargetFrame, Maps...>& lhs,
                const CoordinateMap<SourceFrame, TargetFrame, Maps...>& rhs) {
  return not(lhs == rhs);
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
auto make_coordinate_map(Maps&&... maps)
    -> CoordinateMap<SourceFrame, TargetFrame, std::decay_t<Maps>...> {
  return CoordinateMap<SourceFrame, TargetFrame, std::decay_t<Maps>...>(
      std::forward<Maps>(maps)...);
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
auto make_coordinate_map_base(Maps&&... maps)
    -> std::unique_ptr<CoordinateMapBase<
        SourceFrame, TargetFrame,
        CoordinateMap<SourceFrame, TargetFrame, std::decay_t<Maps>...>::dim>> {
  return std::make_unique<
      CoordinateMap<SourceFrame, TargetFrame, std::decay_t<Maps>...>>(
      std::forward<Maps>(maps)...);
}

template <typename SourceFrame, typename TargetFrame, typename Arg0,
          typename... Args>
auto make_vector_coordinate_map_base(Arg0&& arg_0, Args&&... remaining_args)
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
                                     const Maps&... remaining_maps)
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
    std::index_sequence<Is...> /*meta*/) {
  return CoordinateMap<SourceFrame, TargetFrame, Maps..., NewMap>{
      std::move(std::get<Is>(old_map.maps_))..., std::move(new_map)};
}

template <typename... NewMaps, typename SourceFrame, typename TargetFrame,
          typename... Maps, size_t... Is, size_t... Js>
CoordinateMap<SourceFrame, TargetFrame, Maps..., NewMaps...> push_back_impl(
    CoordinateMap<SourceFrame, TargetFrame, Maps...>&& old_map,
    CoordinateMap<SourceFrame, TargetFrame, NewMaps...> new_map,
    std::index_sequence<Is...> /*meta*/, std::index_sequence<Js...> /*meta*/) {
  return CoordinateMap<SourceFrame, TargetFrame, Maps..., NewMaps...>{
      std::move(std::get<Is>(old_map.maps_))...,
      std::move(std::get<Js>(new_map.maps_))...};
}

template <typename NewMap, typename SourceFrame, typename TargetFrame,
          typename... Maps, size_t... Is>
CoordinateMap<SourceFrame, TargetFrame, NewMap, Maps...> push_front_impl(
    CoordinateMap<SourceFrame, TargetFrame, Maps...>&& old_map, NewMap new_map,
    std::index_sequence<Is...> /*meta*/) {
  return CoordinateMap<SourceFrame, TargetFrame, Maps..., NewMap>{
      std::move(new_map), std::move(std::get<Is>(old_map.maps_))...};
}

template <typename SourceFrame, typename TargetFrame, typename... Maps,
          typename NewMap>
CoordinateMap<SourceFrame, TargetFrame, Maps..., NewMap> push_back(
    CoordinateMap<SourceFrame, TargetFrame, Maps...> old_map, NewMap new_map) {
  return push_back_impl(std::move(old_map), std::move(new_map),
                        std::make_index_sequence<sizeof...(Maps)>{});
}

template <typename SourceFrame, typename TargetFrame, typename... Maps,
          typename... NewMaps>
CoordinateMap<SourceFrame, TargetFrame, Maps..., NewMaps...> push_back(
    CoordinateMap<SourceFrame, TargetFrame, Maps...> old_map,
    CoordinateMap<SourceFrame, TargetFrame, NewMaps...> new_map) {
  return push_back_impl(std::move(old_map), std::move(new_map),
                        std::make_index_sequence<sizeof...(Maps)>{},
                        std::make_index_sequence<sizeof...(NewMaps)>{});
}

template <typename SourceFrame, typename TargetFrame, typename... Maps,
          typename NewMap>
CoordinateMap<SourceFrame, TargetFrame, NewMap, Maps...> push_front(
    CoordinateMap<SourceFrame, TargetFrame, Maps...> old_map, NewMap new_map) {
  return push_front_impl(std::move(old_map), std::move(new_map),
                         std::make_index_sequence<sizeof...(Maps)>{});
}

template <typename SourceFrame, typename TargetFrame, typename... Maps,
          typename... NewMaps>
CoordinateMap<SourceFrame, TargetFrame, NewMaps..., Maps...> push_front(
    CoordinateMap<SourceFrame, TargetFrame, Maps...> old_map,
    CoordinateMap<SourceFrame, TargetFrame, NewMaps...> new_map) {
  return push_back_impl(std::move(new_map), std::move(old_map),
                        std::make_index_sequence<sizeof...(NewMaps)>{},
                        std::make_index_sequence<sizeof...(Maps)>{});
}
}  // namespace domain
/// \endcond
