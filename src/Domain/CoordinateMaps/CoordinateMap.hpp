// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class CoordinateMap

#pragma once

#include <algorithm>
#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <pup.h>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "DataStructures/Tensor/Identity.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/Tuple.hpp"

/// \cond
class DataVector;
class FunctionOfTime;
/// \endcond

namespace domain {
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
  static constexpr size_t dim = Dim;
  using source_frame = SourceFrame;
  using target_frame = TargetFrame;

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
      tnsr::I<double, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list =
          std::unordered_map<std::string, FunctionOfTime&>{}) const
      noexcept = 0;
  virtual tnsr::I<DataVector, Dim, TargetFrame> operator()(
      tnsr::I<DataVector, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list =
          std::unordered_map<std::string, FunctionOfTime&>{}) const
      noexcept = 0;
  // @}

  // @{
  /// Apply the inverse `Maps` to the point(s) `target_point`.
  /// The returned boost::optional is invalid if the map is not invertible
  /// at `target_point`, or if `target_point` can be easily determined to not
  /// make sense for the map.  An example of the latter is passing a
  /// point with a negative value of z into a positive-z Wedge3D inverse map.
  virtual boost::optional<tnsr::I<double, Dim, SourceFrame>> inverse(
      tnsr::I<double, Dim, TargetFrame> target_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list =
          std::unordered_map<std::string, FunctionOfTime&>{}) const
      noexcept = 0;
  // @}

  // @{
  /// Compute the inverse Jacobian of the `Maps` at the point(s)
  /// `source_point`
  virtual InverseJacobian<double, Dim, SourceFrame, TargetFrame> inv_jacobian(
      tnsr::I<double, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list =
          std::unordered_map<std::string, FunctionOfTime&>{}) const
      noexcept = 0;
  virtual InverseJacobian<DataVector, Dim, SourceFrame, TargetFrame>
  inv_jacobian(
      tnsr::I<DataVector, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list =
          std::unordered_map<std::string, FunctionOfTime&>{}) const
      noexcept = 0;
  // @}

  // @{
  /// Compute the Jacobian of the `Maps` at the point(s) `source_point`
  virtual Jacobian<double, Dim, SourceFrame, TargetFrame> jacobian(
      tnsr::I<double, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list =
          std::unordered_map<std::string, FunctionOfTime&>{}) const
      noexcept = 0;
  virtual Jacobian<DataVector, Dim, SourceFrame, TargetFrame> jacobian(
      tnsr::I<DataVector, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list =
          std::unordered_map<std::string, FunctionOfTime&>{}) const
      noexcept = 0;
  // @}
 private:
  virtual bool is_equal_to(const CoordinateMapBase& other) const = 0;
  friend bool operator==(const CoordinateMapBase& lhs,
                         const CoordinateMapBase& rhs) noexcept {
    return typeid(lhs) == typeid(rhs) and lhs.is_equal_to(rhs);
  }
  friend bool operator!=(const CoordinateMapBase& lhs,
                         const CoordinateMapBase& rhs) noexcept {
    return not(lhs == rhs);
  }
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
      tnsr::I<double, dim, SourceFrame> source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list =
          std::unordered_map<std::string, FunctionOfTime&>{}) const
      noexcept override {
    return call_impl(std::move(source_point), time, f_of_t_list,
                     std::make_index_sequence<sizeof...(Maps)>{});
  }
  constexpr tnsr::I<DataVector, dim, TargetFrame> operator()(
      tnsr::I<DataVector, dim, SourceFrame> source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list =
          std::unordered_map<std::string, FunctionOfTime&>{}) const
      noexcept override {
    return call_impl(std::move(source_point), time, f_of_t_list,
                     std::make_index_sequence<sizeof...(Maps)>{});
  }
  // @}

  // @{
  /// Apply the inverse `Maps...` to the point(s) `target_point`
  constexpr boost::optional<tnsr::I<double, dim, SourceFrame>> inverse(
      tnsr::I<double, dim, TargetFrame> target_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list =
          std::unordered_map<std::string, FunctionOfTime&>{}) const
      noexcept override {
    return inverse_impl(std::move(target_point), time, f_of_t_list,
                        std::make_index_sequence<sizeof...(Maps)>{});
  }
  // @}

  // @{
  /// Compute the inverse Jacobian of the `Maps...` at the point(s)
  /// `source_point`
  constexpr InverseJacobian<double, dim, SourceFrame, TargetFrame> inv_jacobian(
      tnsr::I<double, dim, SourceFrame> source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list =
          std::unordered_map<std::string, FunctionOfTime&>{}) const
      noexcept override {
    return inv_jacobian_impl(std::move(source_point), time, f_of_t_list);
  }
  constexpr InverseJacobian<DataVector, dim, SourceFrame, TargetFrame>
  inv_jacobian(
      tnsr::I<DataVector, dim, SourceFrame> source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list =
          std::unordered_map<std::string, FunctionOfTime&>{}) const
      noexcept override {
    return inv_jacobian_impl(std::move(source_point), time, f_of_t_list);
  }
  // @}

  // @{
  /// Compute the Jacobian of the `Maps...` at the point(s) `source_point`
  constexpr Jacobian<double, dim, SourceFrame, TargetFrame> jacobian(
      tnsr::I<double, dim, SourceFrame> source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list =
          std::unordered_map<std::string, FunctionOfTime&>{}) const
      noexcept override {
    return jacobian_impl(std::move(source_point), time, f_of_t_list);
  }
  constexpr Jacobian<DataVector, dim, SourceFrame, TargetFrame> jacobian(
      tnsr::I<DataVector, dim, SourceFrame> source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list =
          std::unordered_map<std::string, FunctionOfTime&>{}) const
      noexcept override {
    return jacobian_impl(std::move(source_point), time, f_of_t_list);
  }
  // @}

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(CoordinateMapBase<SourceFrame, TargetFrame, dim>),
      CoordinateMap);

  explicit CoordinateMap(CkMigrateMessage* /*unused*/) {}

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) override {  // NOLINT
    CoordinateMapBase<SourceFrame, TargetFrame, dim>::pup(p);
    PUP::pup(p, maps_);
  }

 private:
  friend bool operator==(const CoordinateMap& lhs,
                         const CoordinateMap& rhs) noexcept {
    return lhs.maps_ == rhs.maps_;
  }

  bool is_equal_to(const CoordinateMapBase<SourceFrame, TargetFrame, dim>&
                       other) const override {
    const auto& cast_of_other = dynamic_cast<const CoordinateMap&>(other);
    return *this == cast_of_other;
  }

  template <typename T, size_t... Is>
  constexpr SPECTRE_ALWAYS_INLINE tnsr::I<T, dim, TargetFrame> call_impl(
      tnsr::I<T, dim, SourceFrame>&& source_point, double time,
      const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list,
      std::index_sequence<Is...> /*meta*/) const noexcept;

  template <typename T, size_t... Is>
  constexpr SPECTRE_ALWAYS_INLINE boost::optional<tnsr::I<T, dim, SourceFrame>>
  inverse_impl(
      tnsr::I<T, dim, TargetFrame>&& target_point, double time,
      const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list,
      std::index_sequence<Is...> /*meta*/) const noexcept;

  template <typename T>
  constexpr SPECTRE_ALWAYS_INLINE InverseJacobian<T, dim, SourceFrame,
                                                  TargetFrame>
  inv_jacobian_impl(
      tnsr::I<T, dim, SourceFrame>&& source_point, double time,
      const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list) const
      noexcept;

  template <typename T>
  constexpr SPECTRE_ALWAYS_INLINE Jacobian<T, dim, SourceFrame, TargetFrame>
  jacobian_impl(
      tnsr::I<T, dim, SourceFrame>&& source_point, double time,
      const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list) const
      noexcept;

  std::tuple<Maps...> maps_;
};

////////////////////////////////////////////////////////////////
// CoordinateMap definitions
////////////////////////////////////////////////////////////////

// define type-trait to check for time-dependent mapping
namespace CoordinateMap_detail {
template <typename T>
using is_map_time_dependent_t =
    tt::is_callable_t<T, std::array<std::decay_t<T>, std::decay_t<T>::dim>,
                      double, std::unordered_map<std::string, FunctionOfTime&>>;
}  // namespace CoordinateMap_detail

template <typename SourceFrame, typename TargetFrame, typename... Maps>
constexpr CoordinateMap<SourceFrame, TargetFrame, Maps...>::CoordinateMap(
    Maps... maps)
    : maps_(std::move(maps)...) {}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
template <typename T, size_t... Is>
constexpr SPECTRE_ALWAYS_INLINE tnsr::I<
    T, CoordinateMap<SourceFrame, TargetFrame, Maps...>::dim, TargetFrame>
CoordinateMap<SourceFrame, TargetFrame, Maps...>::call_impl(
    tnsr::I<T, dim, SourceFrame>&& source_point, const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list,
    std::index_sequence<Is...> /*meta*/) const noexcept {
  std::array<T, dim> mapped_point = make_array<T, dim>(std::move(source_point));

  (void)std::initializer_list<char>{make_overloader(
      [](const auto& the_map, std::array<T, dim>& point, const double /*t*/,
         const std::unordered_map<std::string, FunctionOfTime&>&
         /*f_of_ts*/,
         const std::false_type /*is_time_independent*/) noexcept {
        if (LIKELY(not the_map.is_identity())) {
          point = the_map(point);
        }
        return '0';
      },
      [](const auto& the_map, std::array<T, dim>& point, const double t,
         const std::unordered_map<std::string, FunctionOfTime&>& f_of_ts,
         const std::true_type /*is_time_dependent*/) noexcept {
        point = the_map(point, t, f_of_ts);
        return '0';
      })(std::get<Is>(maps_), mapped_point, time, f_of_t_list,
         CoordinateMap_detail::is_map_time_dependent_t<Maps>{})...};

  return tnsr::I<T, dim, TargetFrame>(std::move(mapped_point));
}

template <typename SourceFrame, typename TargetFrame, typename... Maps>
template <typename T, size_t... Is>
constexpr SPECTRE_ALWAYS_INLINE boost::optional<tnsr::I<
    T, CoordinateMap<SourceFrame, TargetFrame, Maps...>::dim, SourceFrame>>
CoordinateMap<SourceFrame, TargetFrame, Maps...>::inverse_impl(
    tnsr::I<T, dim, TargetFrame>&& target_point, const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list,
    std::index_sequence<Is...> /*meta*/) const noexcept {
  boost::optional<std::array<T, dim>> mapped_point(
      make_array<T, dim>(std::move(target_point)));

  (void)std::initializer_list<char>{make_overloader(
      [](const auto& the_map, boost::optional<std::array<T, dim>>& point,
         const double /*t*/,
         const std::unordered_map<std::string, FunctionOfTime&>&
         /*f_of_ts*/,
         const std::false_type /*is_time_independent*/) noexcept {
        if (point) {
          if (LIKELY(not the_map.is_identity())) {
            point = the_map.inverse(point.get());
          }
        }
        return '0';
      },
      [](const auto& the_map, boost::optional<std::array<T, dim>>& point,
         const double t,
         const std::unordered_map<std::string, FunctionOfTime&>& f_of_ts,
         const std::true_type /*is_time_dependent*/) noexcept {
        if (point) {
          point = the_map.inverse(point.get(), t, f_of_ts);
        }
        return '0';
        // this is the inverse function, so the iterator sequence below is
        // reversed
      })(std::get<sizeof...(Maps) - 1 - Is>(maps_), mapped_point, time,
         f_of_t_list,
         CoordinateMap_detail::is_map_time_dependent_t<Maps>{})...};

  return mapped_point
             ? tnsr::I<T, dim, SourceFrame>(std::move(mapped_point.get()))
             : boost::optional<tnsr::I<T, dim, SourceFrame>>{};
}

// define type-trait to check for time-dependent jacobian
namespace CoordinateMap_detail {
CREATE_IS_CALLABLE(jacobian)
template <typename Map, typename T>
using is_jacobian_time_dependent_t =
    CoordinateMap_detail::is_jacobian_callable_t<
        Map, std::array<std::decay_t<T>, std::decay_t<Map>::dim>, double,
        std::unordered_map<std::string, FunctionOfTime&>>;
}  // namespace CoordinateMap_detail

template <typename SourceFrame, typename TargetFrame, typename... Maps>
template <typename T>
constexpr SPECTRE_ALWAYS_INLINE auto
CoordinateMap<SourceFrame, TargetFrame, Maps...>::inv_jacobian_impl(
    tnsr::I<T, dim, SourceFrame>&& source_point, const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list) const
    noexcept -> InverseJacobian<T, dim, SourceFrame, TargetFrame> {
  std::array<T, dim> mapped_point = make_array<T, dim>(std::move(source_point));

  InverseJacobian<T, dim, SourceFrame, TargetFrame> inv_jac{};

  tuple_transform(
      maps_,
      [&inv_jac, &mapped_point, time, &f_of_t_list](
          const auto& map, auto index, const std::tuple<Maps...>& maps) {
        constexpr const size_t count = decltype(index)::value;

        tnsr::Ij<T, dim, Frame::NoFrame> temp_inv_jac{};

        // chooses the correct call based on time-dependence of jacobian
        auto inv_jac_overload = make_overloader(
            [](const gsl::not_null<tnsr::Ij<T, dim, Frame::NoFrame>*> t_inv_jac,
               const auto& the_map, const std::array<T, dim>& point,
               const double /*t*/,
               const std::unordered_map<std::string, FunctionOfTime&>&
               /*f_of_ts*/,
               const std::false_type /*is_time_independent*/) {
              if (LIKELY(not the_map.is_identity())) {
                *t_inv_jac = the_map.inv_jacobian(point);
              } else {
                *t_inv_jac = identity<dim>(point[0]);
              }
              return nullptr;
            },
            [](const gsl::not_null<tnsr::Ij<T, dim, Frame::NoFrame>*> t_inv_jac,
               const auto& the_map, const std::array<T, dim>& point,
               const double t,
               const std::unordered_map<std::string, FunctionOfTime&>& f_of_ts,
               const std::true_type /*is_time_dependent*/) {
              *t_inv_jac = the_map.inv_jacobian(point, t, f_of_ts);
              return nullptr;
            });

        if (LIKELY(count != 0)) {
          const auto& map_in_loop =
              std::get<(count != 0 ? count - 1 : 0)>(maps);
          if (LIKELY(not map_in_loop.is_identity())) {
            mapped_point = map_in_loop(mapped_point);
            inv_jac_overload(&temp_inv_jac, map, mapped_point, time,
                             f_of_t_list,
                             CoordinateMap_detail::is_jacobian_time_dependent_t<
                                 decltype(map), T>{});
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
          }
        } else {
          inv_jac_overload(
              &temp_inv_jac, map, mapped_point, time, f_of_t_list,
              CoordinateMap_detail::is_jacobian_time_dependent_t<decltype(map),
                                                                 T>{});
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
    tnsr::I<T, dim, SourceFrame>&& source_point, const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& f_of_t_list) const
    noexcept -> Jacobian<T, dim, SourceFrame, TargetFrame> {
  std::array<T, dim> mapped_point = make_array<T, dim>(std::move(source_point));
  Jacobian<T, dim, SourceFrame, TargetFrame> jac{};

  tuple_transform(
      maps_,
      [&jac, &mapped_point, time, &f_of_t_list](
          const auto& map, auto index, const std::tuple<Maps...>& maps) {
        constexpr const size_t count = decltype(index)::value;

        tnsr::Ij<T, dim, Frame::NoFrame> noframe_jac{};

        // chooses the correct call based on time-dependence of jacobian
        auto jac_overload = make_overloader(
            [](const gsl::not_null<tnsr::Ij<T, dim, Frame::NoFrame>*>
                   no_frame_jac,
               const auto& the_map, const std::array<T, dim>& point,
               const double /*t*/,
               const std::unordered_map<std::string, FunctionOfTime&>&
               /*f_of_ts*/,
               const std::false_type /*is_time_independent*/) {
              if (LIKELY(not the_map.is_identity())) {
                *no_frame_jac = the_map.jacobian(point);
              } else {
                *no_frame_jac = identity<dim>(point[0]);
              }
              return nullptr;
            },
            [](const gsl::not_null<tnsr::Ij<T, dim, Frame::NoFrame>*>
                   no_frame_jac,
               const auto& the_map, const std::array<T, dim>& point,
               const double t,
               const std::unordered_map<std::string, FunctionOfTime&>& f_of_ts,
               const std::true_type /*is_time_dependent*/) {
              *no_frame_jac = the_map.jacobian(point, t, f_of_ts);
              return nullptr;
            });

        if (LIKELY(count != 0)) {
          const auto& map_in_loop =
              std::get<(count != 0 ? count - 1 : 0)>(maps);
          if (LIKELY(not map_in_loop.is_identity())) {
            mapped_point = map_in_loop(mapped_point);
            jac_overload(&noframe_jac, map, mapped_point, time, f_of_t_list,
                         CoordinateMap_detail::is_jacobian_time_dependent_t<
                             decltype(map), T>{});
            std::array<T, dim> temp{};
            for (size_t source = 0; source < dim; ++source) {
              for (size_t target = 0; target < dim; ++target) {
                gsl::at(temp, target) =
                    noframe_jac.get(target, 0) * jac.get(0, source);
                for (size_t dummy = 1; dummy < dim; ++dummy) {
                  gsl::at(temp, target) +=
                      noframe_jac.get(target, dummy) * jac.get(dummy, source);
                }
              }
              for (size_t target = 0; target < dim; ++target) {
                jac.get(target, source) = std::move(gsl::at(temp, target));
              }
            }
          }
        } else {
          jac_overload(
              &noframe_jac, map, mapped_point, time, f_of_t_list,
              CoordinateMap_detail::is_jacobian_time_dependent_t<decltype(map),
                                                                 T>{});
          for (size_t target = 0; target < dim; ++target) {
            for (size_t source = 0; source < dim; ++source) {
              jac.get(target, source) =
                  std::move(noframe_jac.get(target, source));
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

/// \ingroup ComputationalDomainGroup
/// \brief Creates a `CoordinateMap` of `maps...`
template <typename SourceFrame, typename TargetFrame, typename... Maps>
constexpr auto make_coordinate_map(Maps&&... maps)
    -> CoordinateMap<SourceFrame, TargetFrame, std::decay_t<Maps>...> {
  return CoordinateMap<SourceFrame, TargetFrame, std::decay_t<Maps>...>(
      std::forward<Maps>(maps)...);
}

/// \ingroup ComputationalDomainGroup
/// \brief Creates a `std::unique_ptr<CoordinateMapBase>` of `maps...`
template <typename SourceFrame, typename TargetFrame, typename... Maps>
auto make_coordinate_map_base(Maps&&... maps) noexcept
    -> std::unique_ptr<CoordinateMapBase<
        SourceFrame, TargetFrame,
        CoordinateMap<SourceFrame, TargetFrame, std::decay_t<Maps>...>::dim>> {
  return std::make_unique<
      CoordinateMap<SourceFrame, TargetFrame, std::decay_t<Maps>...>>(
      std::forward<Maps>(maps)...);
}

/// \ingroup ComputationalDomainGroup
/// \brief Creates a `std::vector<std::unique_ptr<CoordinateMapBase>>`
/// containing the result of `make_coordinate_map_base` applied to each
/// argument passed in.
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
  (void)std::initializer_list<int>{
      (((void)return_vector.emplace_back(
           make_coordinate_map_base<SourceFrame, TargetFrame>(
               std::forward<Args>(remaining_args)))),
       0)...};
  return return_vector;
}

/// \ingroup ComputationalDomainGroup
/// \brief Creates a `std::vector<std::unique_ptr<CoordinateMapBase>>`
/// containing the result of `make_coordinate_map_base` applied to each
/// element of the vector of maps composed with the rest of the arguments
/// passed in.
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

/// \cond
template <typename SourceFrame, typename TargetFrame, typename... Maps>
PUP::able::PUP_ID
    CoordinateMap<SourceFrame, TargetFrame, Maps...>::my_PUP_ID =  // NOLINT
    0;
/// \endcond
}  // namespace domain
