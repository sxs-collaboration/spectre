// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class CoordinateMap

#pragma once

#include <memory>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Tuple.hpp"

/// Contains all coordinate maps.
namespace CoordinateMaps {
template <typename FirstMap, typename... Maps>
constexpr size_t map_dim = FirstMap::dim;
}  // namespace CoordinateMaps

/*!
 * \ingroup CoordinateMapsGroup
 * \brief Abstract base class for CoordinateMap
 */
template <typename SourceFrame, typename TargetFrame, size_t Dim>
class CoordinateMapBase : public PUP::able {
 public:
  WRAPPED_PUPable_abstract(CoordinateMapBase);  // NOLINT

  CoordinateMapBase() = default;
  CoordinateMapBase(const CoordinateMapBase& /*rhs*/) = default;
  CoordinateMapBase& operator=(const CoordinateMapBase& /*rhs*/) = default;
  CoordinateMapBase(CoordinateMapBase&& /*rhs*/) = default;
  CoordinateMapBase& operator=(CoordinateMapBase&& /*rhs*/) = default;
  ~CoordinateMapBase() override = default;

  virtual std::unique_ptr<CoordinateMapBase<SourceFrame, TargetFrame, Dim>>
  get_clone() const = 0;

  // @{
  /// Apply the `Maps` to the point(s) `source_point`
  virtual tnsr::I<double, Dim, TargetFrame> operator()(
      const tnsr::I<double, Dim, SourceFrame>& source_point) const = 0;
  virtual tnsr::I<DataVector, Dim, TargetFrame> operator()(
      const tnsr::I<DataVector, Dim, SourceFrame>& source_point) const = 0;
  // @}

  // @{
  /// Apply the inverse `Maps` to the point(s) `target_point`
  virtual tnsr::I<double, Dim, SourceFrame> inverse(
      const tnsr::I<double, Dim, TargetFrame>& target_point) const = 0;
  virtual tnsr::I<DataVector, Dim, SourceFrame> inverse(
      const tnsr::I<DataVector, Dim, TargetFrame>& target_point) const = 0;
  // @}

  // @{
  /// Compute the inverse Jacobian of the `Maps` at the point(s)
  /// `source_point`
  virtual Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                 index_list<SpatialIndex<Dim, UpLo::Up, SourceFrame>,
                            SpatialIndex<Dim, UpLo::Lo, TargetFrame>>>
  inv_jacobian(const tnsr::I<double, Dim, SourceFrame>& source_point) const = 0;
  virtual Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                 index_list<SpatialIndex<Dim, UpLo::Up, SourceFrame>,
                            SpatialIndex<Dim, UpLo::Lo, TargetFrame>>>
  inv_jacobian(
      const tnsr::I<DataVector, Dim, SourceFrame>& source_point) const = 0;
  // @}

  // @{
  /// Compute the Jacobian of the `Maps` at the point(s) `source_point`
  virtual Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                 index_list<SpatialIndex<Dim, UpLo::Up, TargetFrame>,
                            SpatialIndex<Dim, UpLo::Lo, SourceFrame>>>
  jacobian(const tnsr::I<double, Dim, SourceFrame>& source_point) const = 0;
  virtual Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                 index_list<SpatialIndex<Dim, UpLo::Up, TargetFrame>,
                            SpatialIndex<Dim, UpLo::Lo, SourceFrame>>>
  jacobian(const tnsr::I<DataVector, Dim, SourceFrame>& source_point) const = 0;
  // @}
};

/*!
 * \ingroup CoordinateMapsGroup
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
class CoordinateMap
    : public CoordinateMapBase<SourceFrame, TargetFrame,
                               CoordinateMaps::map_dim<Maps...>> {
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

  CoordinateMap(const CoordinateMap& /*rhs*/) = default;
  CoordinateMap& operator=(const CoordinateMap& /*rhs*/) = default;
  CoordinateMap(CoordinateMap&& /*rhs*/) = default;
  CoordinateMap& operator=(CoordinateMap&& /*rhs*/) = default;
  ~CoordinateMap() override = default;

  constexpr explicit CoordinateMap(Maps... maps);

  std::unique_ptr<CoordinateMapBase<SourceFrame, TargetFrame, dim>> get_clone()
      const override {
    return std::make_unique<CoordinateMap>(*this);
  }

  // @{
  /// Apply the `Maps...` to the point(s) `source_point`
  constexpr tnsr::I<double, dim, TargetFrame> operator()(
      const tnsr::I<double, dim, SourceFrame>& source_point) const override {
    return call_impl(source_point);
  }
  constexpr tnsr::I<DataVector, dim, TargetFrame> operator()(
      const tnsr::I<DataVector, dim, SourceFrame>& source_point)
      const override {
    return call_impl(source_point);
  }
  // @}

  // @{
  /// Apply the inverse `Maps...` to the point(s) `target_point`
  constexpr tnsr::I<double, dim, SourceFrame> inverse(
      const tnsr::I<double, dim, TargetFrame>& target_point) const override {
    return inverse_impl(target_point);
  }
  constexpr tnsr::I<DataVector, dim, SourceFrame> inverse(
      const tnsr::I<DataVector, dim, TargetFrame>& target_point)
      const override {
    return inverse_impl(target_point);
  }
  // @}

  // @{
  /// Compute the inverse Jacobian of the `Maps...` at the point(s)
  /// `source_point`
  constexpr Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                   index_list<SpatialIndex<dim, UpLo::Up, SourceFrame>,
                              SpatialIndex<dim, UpLo::Lo, TargetFrame>>>
  inv_jacobian(
      const tnsr::I<double, dim, SourceFrame>& source_point) const override {
    return inv_jacobian_impl(source_point);
  }
  constexpr Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                   index_list<SpatialIndex<dim, UpLo::Up, SourceFrame>,
                              SpatialIndex<dim, UpLo::Lo, TargetFrame>>>
  inv_jacobian(const tnsr::I<DataVector, dim, SourceFrame>& source_point)
      const override {
    return inv_jacobian_impl(source_point);
  }
  // @}

  // @{
  /// Compute the Jacobian of the `Maps...` at the point(s) `source_point`
  constexpr Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                   index_list<SpatialIndex<dim, UpLo::Up, TargetFrame>,
                              SpatialIndex<dim, UpLo::Lo, SourceFrame>>>
  jacobian(
      const tnsr::I<double, dim, SourceFrame>& source_point) const override {
    return jacobian_impl(source_point);
  }
  constexpr Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                   index_list<SpatialIndex<dim, UpLo::Up, TargetFrame>,
                              SpatialIndex<dim, UpLo::Lo, SourceFrame>>>
  jacobian(const tnsr::I<DataVector, dim, SourceFrame>& source_point)
      const override {
    return jacobian_impl(source_point);
  }
  // @}

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(CoordinateMapBase<SourceFrame, TargetFrame, dim>),
      CoordinateMap);

  explicit CoordinateMap(CkMigrateMessage* /*unused*/) {}

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) override {  // NOLINT
    CoordinateMapBase<SourceFrame, TargetFrame, dim>::pup(p);
    p | maps_;
  }

 private:
  friend bool operator==(const CoordinateMap& lhs,
                         const CoordinateMap& rhs) noexcept {
    return lhs.maps_ == rhs.maps_;
  }

  template <typename T>
  constexpr SPECTRE_ALWAYS_INLINE tnsr::I<T, dim, TargetFrame> call_impl(
      const tnsr::I<T, dim, SourceFrame>& source_point) const;

  template <typename T>
  constexpr SPECTRE_ALWAYS_INLINE tnsr::I<T, dim, SourceFrame> inverse_impl(
      const tnsr::I<T, dim, TargetFrame>& target_point) const;

  template <typename T>
  constexpr SPECTRE_ALWAYS_INLINE
      Tensor<T, tmpl::integral_list<std::int32_t, 2, 1>,
             index_list<SpatialIndex<dim, UpLo::Up, SourceFrame>,
                        SpatialIndex<dim, UpLo::Lo, TargetFrame>>>
      inv_jacobian_impl(const tnsr::I<T, dim, SourceFrame>& source_point) const;

  template <typename T>
  constexpr SPECTRE_ALWAYS_INLINE
      Tensor<T, tmpl::integral_list<std::int32_t, 2, 1>,
             index_list<SpatialIndex<dim, UpLo::Up, TargetFrame>,
                        SpatialIndex<dim, UpLo::Lo, SourceFrame>>>
      jacobian_impl(const tnsr::I<T, dim, SourceFrame>& source_point) const;

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
constexpr SPECTRE_ALWAYS_INLINE tnsr::I<
    T, CoordinateMap<SourceFrame, TargetFrame, Maps...>::dim, TargetFrame>
CoordinateMap<SourceFrame, TargetFrame, Maps...>::call_impl(
    const tnsr::I<T, dim, SourceFrame>& source_point) const {
  std::array<T, dim> mapped_point = make_array<T, dim>(source_point);
  tuple_fold(maps_, [](const auto& map,
                       std::array<T, dim>& point) { point = map(point); },
             mapped_point);
  return tnsr::I<T, dim, TargetFrame>(std::move(mapped_point));
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
template <typename T>
constexpr SPECTRE_ALWAYS_INLINE tnsr::I<
    T, CoordinateMap<SourceFrame, TargetFrame, Maps...>::dim, SourceFrame>
CoordinateMap<SourceFrame, TargetFrame, Maps...>::inverse_impl(
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
constexpr SPECTRE_ALWAYS_INLINE auto
CoordinateMap<SourceFrame, TargetFrame, Maps...>::inv_jacobian_impl(
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
constexpr SPECTRE_ALWAYS_INLINE auto
CoordinateMap<SourceFrame, TargetFrame, Maps...>::jacobian_impl(
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
constexpr CoordinateMap<SourceFrame, TargetFrame, std::decay_t<Maps>...>
make_coordinate_map(Maps&&... maps) {
  return CoordinateMap<SourceFrame, TargetFrame, std::decay_t<Maps>...>(
      std::forward<Maps>(maps)...);
}

/// \cond
template <typename SourceFrame, typename TargetFrame, typename... Maps>
PUP::able::PUP_ID
    CoordinateMap<SourceFrame, TargetFrame, Maps...>::my_PUP_ID =  // NOLINT
    0;
/// \endcond
