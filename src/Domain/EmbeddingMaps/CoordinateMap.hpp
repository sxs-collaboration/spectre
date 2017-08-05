// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class CoordinateMap

#pragma once

#include <memory>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArithmeticValue.hpp"
#include "Utilities/Tuple.hpp"

/// Contains all embedding maps.
namespace CoordinateMaps {
template <typename FirstMap, typename... Maps>
constexpr size_t map_dim = FirstMap::dim;
}  // namespace CoordinateMaps

/*!
 * \ingroup ComputationalDomain
 * \brief A coordinate map or composition of coordinate maps
 *
 * Maps coordinates from the `SourceFrame` to the `TargetFrame` using the
 * coordinate maps `Maps...`. The individual maps are applied left to right
 * from the source to the target Frame. The inverse map, as well as Jacobian
 * and inverse Jacobian are also provided. The `CoordinateMap` class must
 * be used even if just wrapping a single coordinate map. It is designed to
 * be an extremely minimal interface to the underlying coordinate maps. For
 * a list of all coordinate maps see the CoordinateMaps group or namespace.
 *
 * Each coordinate map must contain a `static constexpr size_t dim` variable
 * that is equal to the dimensionality of the map. The Coordinatemap class
 * contains a member `static constexpr size_t dim`, a type alias `source_frame`,
 * a type alias `target_frame` and `typelist of the `Maps...`.
 */
template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap {
  static_assert(sizeof...(Maps) > 0, "Must have at least one map");
  static_assert(
      tmpl::all<tmpl::integral_list<size_t, Maps::dim...>,
                std::is_same<tmpl::integral_constant<
                                 size_t, CoordinateMaps::map_dim<Maps...>>,
                             tmpl::_1>>::value,
      "All Maps passed to CoordinateMap must be of the same dimensionality.");

 public:
  static constexpr size_t dim = CoordinateMaps::map_dim<Maps...>;
  using source_frame = SourceFrame;
  using target_frame = TargetFrame;
  using maps_list = tmpl::list<Maps...>;

  /// Used for Charm++ serialization
  CoordinateMap() = default;

  constexpr explicit CoordinateMap(Maps... maps);

  /// Apply the `Maps...` to the point(s) `source_point`
  template <typename T>
  constexpr tnsr::I<T, dim, TargetFrame> operator()(
      const tnsr::I<T, dim, SourceFrame>& source_point) const;

  /// Apply the inverse `Maps...` to the point(s) `target_point`
  template <typename T>
  constexpr tnsr::I<T, dim, SourceFrame> inverse(
      const tnsr::I<T, dim, TargetFrame>& target_point) const;

  /// Compute the inverse Jacobian of the `Maps...` at the point(s)
  /// `source_point`
  template <typename T>
  constexpr Tensor<T, tmpl::integral_list<std::int32_t, 2, 1>,
                   index_list<SpatialIndex<dim, UpLo::Up, SourceFrame>,
                              SpatialIndex<dim, UpLo::Lo, TargetFrame>>>
  inv_jacobian(const tnsr::I<T, dim, SourceFrame>& source_point) const;

  /// Compute the Jacobian of the `Maps...` at the point(s) `source_point`
  template <typename T>
  constexpr Tensor<T, tmpl::integral_list<std::int32_t, 2, 1>,
                   index_list<SpatialIndex<dim, UpLo::Up, TargetFrame>,
                              SpatialIndex<dim, UpLo::Lo, SourceFrame>>>
  jacobian(const tnsr::I<T, dim, SourceFrame>& source_point) const;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) { p | maps_; }  // NOLINT

 private:
  friend bool operator==(const CoordinateMap& lhs,
                         const CoordinateMap& rhs) noexcept {
    return lhs.maps_ == rhs.maps_;
  }

  std::tuple<Maps...> maps_;
};

////////////////////////////////////////////////////////////////
// CoordinateMap definitions
////////////////////////////////////////////////////////////////

template <typename SourceFrame, typename TargetFrame, typename... Maps>
constexpr CoordinateMap<SourceFrame, TargetFrame, Maps...>::CoordinateMap(
    Maps... maps)
    : maps_(std::move(maps)...) {}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
template <typename T>
constexpr tnsr::I<T, CoordinateMap<SourceFrame, TargetFrame, Maps...>::dim,
                  TargetFrame>
CoordinateMap<SourceFrame, TargetFrame, Maps...>::operator()(
    const tnsr::I<T, dim, SourceFrame>& source_point) const {
  std::array<T, dim> mapped_point = make_array<T, dim>(source_point);
  tuple_fold(maps_, [](const auto& map,
                       std::array<T, dim>& point) { point = map(point); },
             mapped_point);
  return tnsr::I<T, dim, TargetFrame>(std::move(mapped_point));
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
template <typename T>
constexpr tnsr::I<T, CoordinateMap<SourceFrame, TargetFrame, Maps...>::dim,
                  SourceFrame>
CoordinateMap<SourceFrame, TargetFrame, Maps...>::inverse(
    const tnsr::I<T, dim, TargetFrame>& target_point) const {
  std::array<T, dim> mapped_point = make_array<T, dim>(target_point);
  tuple_fold<true>(maps_,
                   [](const auto& map, std::array<T, dim>& point) {
                     point = map.inverse(point);
                   },
                   mapped_point);
  return tnsr::I<T, dim, SourceFrame>(std::move(mapped_point));
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
template <typename T>
constexpr auto CoordinateMap<SourceFrame, TargetFrame, Maps...>::inv_jacobian(
    const tnsr::I<T, dim, SourceFrame>& source_point) const
    -> Tensor<T, tmpl::integral_list<std::int32_t, 2, 1>,
              index_list<SpatialIndex<dim, UpLo::Up, SourceFrame>,
                         SpatialIndex<dim, UpLo::Lo, TargetFrame>>> {
  std::array<T, dim> mapped_point = make_array<T, dim>(source_point);

  Tensor<T, tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<dim, UpLo::Up, SourceFrame>,
                    SpatialIndex<dim, UpLo::Lo, TargetFrame>>>
      inv_jac{};

  tuple_transform(
      maps_,
      [&inv_jac, &mapped_point](const auto& map, auto index,
                                const std::tuple<Maps...>& maps) {
        constexpr const size_t count = decltype(index)::value;
        auto temp_inv_jac = map.inv_jacobian(mapped_point);

        if (LIKELY(count != 0)) {
          mapped_point =
              std::get<(count != 0 ? count - 1 : 0)>(maps)(mapped_point);
          std::array<T, dim> temp{};
          for (size_t source = 0; source < dim; ++source) {
            for (size_t target = 0; target < dim; ++target) {
              gsl::at(temp, target) =
                  inv_jac.get(source, 0) * temp_inv_jac.get(0, target);
              for (size_t dummy = 1; dummy < dim; ++dummy) {
                gsl::at(temp, target) += inv_jac.get(source, dummy) *
                                         temp_inv_jac.get(dummy, target);
              }
            }
            for (size_t target = 0; target < dim; ++target) {
              inv_jac.get(source, target) = std::move(gsl::at(temp, target));
            }
          }
        } else {
          for (size_t source = 0; source < dim; ++source) {
            for (size_t target = 0; target < dim; ++target) {
              inv_jac.get(source, target) =
                  std::move(temp_inv_jac.get(source, target));
            }
          }
        }
      },
      maps_);
  return inv_jac;
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
template <typename T>
constexpr auto CoordinateMap<SourceFrame, TargetFrame, Maps...>::jacobian(
    const tnsr::I<T, dim, SourceFrame>& source_point) const
    -> Tensor<T, tmpl::integral_list<std::int32_t, 2, 1>,
              index_list<SpatialIndex<dim, UpLo::Up, TargetFrame>,
                         SpatialIndex<dim, UpLo::Lo, SourceFrame>>> {
  std::array<T, dim> mapped_point = make_array<T, dim>(source_point);
  Tensor<T, tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<dim, UpLo::Up, TargetFrame>,
                    SpatialIndex<dim, UpLo::Lo, SourceFrame>>>
      jac{};
  tuple_transform<true>(
      maps_,
      [&jac, &mapped_point](const auto& map, auto index,
                            const std::tuple<Maps...>& maps) {
        constexpr const size_t count = decltype(index)::value;
        auto noframe_jac = map.jacobian(mapped_point);

        if (LIKELY(count != sizeof...(Maps) - 1)) {
          mapped_point = std::get<(
              count != sizeof...(Maps) - 1 ? count : sizeof...(Maps) - 1)>(
              maps)(mapped_point);
          std::array<T, dim> temp{};
          for (size_t source = 0; source < dim; ++source) {
            for (size_t target = 0; target < dim; ++target) {
              gsl::at(temp, target) =
                  jac.get(source, 0) * noframe_jac.get(0, target);
              for (size_t dummy = 1; dummy < dim; ++dummy) {
                gsl::at(temp, target) +=
                    jac.get(source, dummy) * noframe_jac.get(dummy, target);
              }
            }
            for (size_t target = 0; target < dim; ++target) {
              jac.get(source, target) = std::move(gsl::at(temp, target));
            }
          }
        } else {
          for (size_t source = 0; source < dim; ++source) {
            for (size_t target = 0; target < dim; ++target) {
              jac.get(source, target) =
                  std::move(noframe_jac.get(source, target));
            }
          }
        }
      },
      maps_);
  return jac;
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
bool operator!=(
    const CoordinateMap<SourceFrame, TargetFrame, Maps...>& lhs,
    const CoordinateMap<SourceFrame, TargetFrame, Maps...>& rhs) noexcept {
  return not(lhs == rhs);
}

/// \ingroup ComputationalDomain
/// \brief Creates a CoordinateMap of `maps...`
template <typename SourceFrame, typename TargetFrame, typename... Maps>
constexpr CoordinateMap<SourceFrame, TargetFrame, Maps...> make_coordinate_map(
    Maps&&... maps) {
  return CoordinateMap<SourceFrame, TargetFrame, Maps...>(
      std::forward<Maps>(maps)...);
}
