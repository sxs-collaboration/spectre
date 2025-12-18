// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class CoordinateMap

#pragma once

#include <cstddef>
#include <limits>
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

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/Autodiff/Autodiff.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Simd/Simd.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"

/// \cond
class DataVector;
/// \endcond

namespace domain {
/// Contains all coordinate maps.
namespace CoordinateMaps {
/// Contains the time-dependent coordinate maps
namespace TimeDependent {}
template <typename FirstMap, typename... Maps>
constexpr size_t map_dim = FirstMap::dim;
}  // namespace CoordinateMaps

namespace CoordinateMap_detail {
CREATE_IS_CALLABLE(function_of_time_names)
CREATE_IS_CALLABLE_V(function_of_time_names)

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

template <typename... Maps>
std::string get_unsupported_autodiff_maps_error() {
  std::string unsupported_maps;
  size_t index = 0;
  (
      [&]() {
        if constexpr (not Maps::supports_hessian) {
          if (not unsupported_maps.empty()) {
            unsupported_maps += ", ";
          }
          unsupported_maps += "Map " + std::to_string(index) + " (" +
                              pretty_type::get_name<Maps>() + ")";
        }
        ++index;
      }(),
      ...);
  return unsupported_maps;
}

}  // namespace CoordinateMap_detail

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

  /// \brief Retrieve the same map but going from `SourceFrame` to
  /// `Frame::Grid`.
  ///
  /// This functionality is needed when composing time-dependent maps with
  /// time-independent maps, where the target frame of the time-independent map
  /// is `Frame::Grid`.
  virtual std::unique_ptr<CoordinateMapBase<SourceFrame, Frame::Grid, Dim>>
  get_to_grid_frame() const = 0;

  /// Returns `true` if the map is the identity
  virtual bool is_identity() const = 0;

  /// Returns `true` if the inverse Jacobian depends on time.
  virtual bool inv_jacobian_is_time_dependent() const = 0;

  /// Returns `true` if the Jacobian depends on time.
  virtual bool jacobian_is_time_dependent() const = 0;

  /// Get a set of all FunctionOfTime names used in this mapping
  virtual const std::unordered_set<std::string>& function_of_time_names()
      const = 0;

  /// Returns `true` if this CoordinateMap supports autodiff
  virtual bool supports_hessian() const = 0;

  /// @{
  /// Apply the `Maps` to the point(s) `source_point`
  virtual tnsr::I<double, Dim, TargetFrame> operator()(
      tnsr::I<double, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const = 0;
  virtual tnsr::I<DataVector, Dim, TargetFrame> operator()(
      tnsr::I<DataVector, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const = 0;
  /// @}

#ifdef SPECTRE_AUTODIFF
  /// @{
  /// Apply the `Maps` to the point(s) `source_point`
  ///
  /// \note Require SPECTRE_AUTODIFF=ON.
  virtual tnsr::I<autodiff::HigherOrderDual<2, double>, Dim, TargetFrame>
  operator()(tnsr::I<autodiff::HigherOrderDual<2, double>, Dim,
                     SourceFrame> /*source_point*/,
             double /*time*/ = std::numeric_limits<double>::signaling_NaN(),
             const FunctionsOfTimeMap& /*functions_of_time*/ = {}) const {
    ERROR("Call operator for autodiff types must be overriden.");
  }
  virtual tnsr::I<autodiff::HigherOrderDual<2, simd::batch<double>>, Dim,
                  TargetFrame>
  operator()(tnsr::I<autodiff::HigherOrderDual<2, simd::batch<double>>, Dim,
                     SourceFrame> /*source_point*/,
             double /*time*/ = std::numeric_limits<double>::signaling_NaN(),
             const FunctionsOfTimeMap& /*functions_of_time*/ = {}) const {
    ERROR("Call operator for autodiff types must be overriden.");
  }
  /// @}
#endif  // SPECTRE_AUTODIFF

  /// @{
  /// Apply the inverse `Maps` to the point(s) `target_point`.
  /// The returned std::optional is invalid if the map is not invertible
  /// at `target_point`, or if `target_point` can be easily determined to not
  /// make sense for the map.  An example of the latter is passing a
  /// point with a negative value of z into a positive-z Wedge<3> inverse map.
  /// The inverse function is only callable with doubles because the inverse
  /// might fail if called for a point out of range, and it is unclear
  /// what should happen if the inverse were to succeed for some points in a
  /// DataVector but fail for other points.
  virtual std::optional<tnsr::I<double, Dim, SourceFrame>> inverse(
      tnsr::I<double, Dim, TargetFrame> target_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const = 0;
  /// @}

  /// @{
  /// Compute the inverse Jacobian of the `Maps` at the point(s)
  /// `source_point`
  virtual InverseJacobian<double, Dim, SourceFrame, TargetFrame> inv_jacobian(
      tnsr::I<double, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const = 0;
  virtual InverseJacobian<DataVector, Dim, SourceFrame, TargetFrame>
  inv_jacobian(tnsr::I<DataVector, Dim, SourceFrame> source_point,
               double time = std::numeric_limits<double>::signaling_NaN(),
               const FunctionsOfTimeMap& functions_of_time = {}) const = 0;
#ifdef SPECTRE_AUTODIFF
  virtual InverseJacobian<autodiff::HigherOrderDual<2, double>, Dim,
                          SourceFrame, TargetFrame>
  inv_jacobian(tnsr::I<autodiff::HigherOrderDual<2, double>, Dim, SourceFrame>
                   source_point,
               double time = std::numeric_limits<double>::signaling_NaN(),
               const FunctionsOfTimeMap& functions_of_time = {}) const = 0;
#endif  // SPECTRE_AUTODIFF
  /// @}

#ifdef SPECTRE_AUTODIFF
  /// @{
  /// Compute the inverse Hessian of the `Maps` at the point(s)
  /// `source_point`
  ///
  /// \note Require SPECTRE_AUTODIFF=ON.
  virtual InverseHessian<double, Dim, SourceFrame, TargetFrame> inv_hessian(
      tnsr::I<double, Dim, SourceFrame> source_point,
      const InverseJacobian<double, Dim, SourceFrame, TargetFrame>& inverse_jac,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const = 0;
  virtual InverseHessian<DataVector, Dim, SourceFrame, TargetFrame> inv_hessian(
      tnsr::I<DataVector, Dim, SourceFrame> source_point,
      const InverseJacobian<DataVector, Dim, SourceFrame, TargetFrame>&
          inverse_jac,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const = 0;
  /// @}
#endif  // SPECTRE_AUTODIFF

  /// @{
  /// Compute the Jacobian of the `Maps` at the point(s) `source_point`
  virtual Jacobian<double, Dim, SourceFrame, TargetFrame> jacobian(
      tnsr::I<double, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const = 0;
  virtual Jacobian<DataVector, Dim, SourceFrame, TargetFrame> jacobian(
      tnsr::I<DataVector, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const = 0;
  /// @}

  /// @{
  /// Compute the mapped coordinates, frame velocity, Jacobian, and inverse
  /// Jacobian
  virtual auto coords_frame_velocity_jacobians(
      tnsr::I<double, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const
      -> std::tuple<tnsr::I<double, Dim, TargetFrame>,
                    InverseJacobian<double, Dim, SourceFrame, TargetFrame>,
                    Jacobian<double, Dim, SourceFrame, TargetFrame>,
                    tnsr::I<double, Dim, TargetFrame>> = 0;
  virtual auto coords_frame_velocity_jacobians(
      tnsr::I<DataVector, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const
      -> std::tuple<tnsr::I<DataVector, Dim, TargetFrame>,
                    InverseJacobian<DataVector, Dim, SourceFrame, TargetFrame>,
                    Jacobian<DataVector, Dim, SourceFrame, TargetFrame>,
                    tnsr::I<DataVector, Dim, TargetFrame>> = 0;
  /// @}

 private:
  virtual bool is_equal_to(const CoordinateMapBase& other) const = 0;
  friend bool operator==(const CoordinateMapBase& lhs,
                         const CoordinateMapBase& rhs) {
    return typeid(lhs) == typeid(rhs) and lhs.is_equal_to(rhs);
  }
  friend bool operator!=(const CoordinateMapBase& lhs,
                         const CoordinateMapBase& rhs) {
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

  explicit CoordinateMap(Maps... maps);

  std::unique_ptr<CoordinateMapBase<SourceFrame, TargetFrame, dim>> get_clone()
      const override {
    return std::make_unique<CoordinateMap>(*this);
  }

  std::unique_ptr<CoordinateMapBase<SourceFrame, Frame::Grid, dim>>
  get_to_grid_frame() const override {
    return get_to_grid_frame_impl(std::make_index_sequence<sizeof...(Maps)>{});
  }

  /// Returns `true` if the map is the identity
  bool is_identity() const override;

  /// Returns `true` if the inverse Jacobian depends on time.
  bool inv_jacobian_is_time_dependent() const override;

  /// Returns `true` if the Jacobian depends on time.
  bool jacobian_is_time_dependent() const override;

  /// Get a set of all FunctionOfTime names from `Maps`
  const std::unordered_set<std::string>& function_of_time_names()
      const override {
    return function_of_time_names_;
  }

  /// Returns `true` if this coordinate map supports hessian
  bool supports_hessian() const override {
    return (Maps::supports_hessian && ...);
  }

  /// @{
  /// Apply the `Maps...` to the point(s) `source_point`
  tnsr::I<double, dim, TargetFrame> operator()(
      tnsr::I<double, dim, SourceFrame> source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const override {
    return call_impl(std::move(source_point), time, functions_of_time,
                     std::make_index_sequence<sizeof...(Maps)>{});
  }
  tnsr::I<DataVector, dim, TargetFrame> operator()(
      tnsr::I<DataVector, dim, SourceFrame> source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const override {
    return call_impl(std::move(source_point), time, functions_of_time,
                     std::make_index_sequence<sizeof...(Maps)>{});
  }
  /// @}

#ifdef SPECTRE_AUTODIFF
  /// @{
  /// Apply the `Maps...` to the point(s) `source_point`
  ///
  /// \note Require SPECTRE_AUTODIFF=ON.
  tnsr::I<autodiff::HigherOrderDual<2, double>, dim, TargetFrame> operator()(
      tnsr::I<autodiff::HigherOrderDual<2, double>, dim, SourceFrame>
          source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const override {
    if constexpr ((Maps::supports_hessian && ...)) {
      return call_impl(std::move(source_point), time, functions_of_time,
                       std::make_index_sequence<sizeof...(Maps)>{});
    } else {
      ERROR("At least one of the Maps does not support autodiff: "
            << CoordinateMap_detail::get_unsupported_autodiff_maps_error<
                   Maps...>());
    }
  }
  tnsr::I<autodiff::HigherOrderDual<2, simd::batch<double>>, dim, TargetFrame>
  operator()(tnsr::I<autodiff::HigherOrderDual<2, simd::batch<double>>, dim,
                     SourceFrame>
                 source_point,
             const double time = std::numeric_limits<double>::signaling_NaN(),
             const FunctionsOfTimeMap& functions_of_time = {}) const override {
    if constexpr ((Maps::supports_hessian && ...)) {
      return call_impl(std::move(source_point), time, functions_of_time,
                       std::make_index_sequence<sizeof...(Maps)>{});
    } else {
      ERROR("At least one of the Maps does not support autodiff: "
            << CoordinateMap_detail::get_unsupported_autodiff_maps_error<
                   Maps...>());
    }
  }
  /// @}
#endif  // SPECTRE_AUTODIFF

  /// @{
  /// Apply the inverse `Maps...` to the point(s) `target_point`
  std::optional<tnsr::I<double, dim, SourceFrame>> inverse(
      tnsr::I<double, dim, TargetFrame> target_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const override {
    return inverse_impl(std::move(target_point), time, functions_of_time,
                        std::make_index_sequence<sizeof...(Maps)>{});
  }
  /// @}

  /// @{
  /// Compute the inverse Jacobian of the `Maps...` at the point(s)
  /// `source_point`
  InverseJacobian<double, dim, SourceFrame, TargetFrame> inv_jacobian(
      tnsr::I<double, dim, SourceFrame> source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const override {
    return inv_jacobian_impl(std::move(source_point), time, functions_of_time);
  }
  InverseJacobian<DataVector, dim, SourceFrame, TargetFrame> inv_jacobian(
      tnsr::I<DataVector, dim, SourceFrame> source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const override {
    return inv_jacobian_impl(std::move(source_point), time, functions_of_time);
  }
#ifdef SPECTRE_AUTODIFF
  InverseJacobian<autodiff::HigherOrderDual<2, double>, dim, SourceFrame,
                  TargetFrame>
  inv_jacobian(
      tnsr::I<autodiff::HigherOrderDual<2, double>, dim, SourceFrame>
          source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const override {
    if constexpr ((Maps::supports_hessian && ...)) {
      return inv_jacobian_impl(std::move(source_point), time,
                               functions_of_time);
    } else {
      ERROR("At least one of the Maps does not support autodiff: "
            << CoordinateMap_detail::get_unsupported_autodiff_maps_error<
                   Maps...>());
    }
  }
#endif  // SPECTRE_AUTODIFF
  /// @}

#ifdef SPECTRE_AUTODIFF
  /// @{
  /*!
   * Compute the inverse Hessian of the `Maps...` at the point(s)
   * `source_point` by computing the Hessian of the `Maps...` and
   * compose it with inverse Jacobians of the `Maps...`.
   *
   * This function propagates autodiff dual types
   * through the `call_impl` function to automatically get the
   * Hessian \f$ \frac{\partial^2 x^i}{\partial\xi^j \partial\xi^k} \f$
   * where \f$ x^i \f$ are the target coordinates and \f$ \xi^i \f$ are the
   * source coordinates. Then the inverse Hessian is computed by the
   * identity
   * \f[\frac{\partial^2 \xi^i}{\partial x^m \partial x^n} =
   * -\frac{\partial \xi^i}{\partial x^j} \frac{\partial \xi^k}{\partial x^m}
   * \frac{\partial \xi^l}{\partial x^n} \frac{\partial^2 x^j}{\partial \xi^k
   * \partial \xi^l}, \f] where the inverse Jacobian is passed in as a function
   * argument.
   *
   * See Test_CoordinateMap.cpp for a different implementation. The current
   * implementation is chosen in production as it is faster when we have
   * the inverse Jacobian already, and in most cases we do.
   *
   * We use forward mode autodiff here as it is simpler to implement and
   * has better optimization than the reverse mode. Reverse mode in
   * the [Autodiff](https://github.com/autodiff/autodiff) library
   * has higher cost per propagation and does not support taping.
   * Also see this
   * [github issue](https://github.com/autodiff/autodiff/issues/332).
   *
   * \note Require SPECTRE_AUTODIFF=ON.
   */
  InverseHessian<double, dim, SourceFrame, TargetFrame> inv_hessian(
      tnsr::I<double, dim, SourceFrame> source_point,
      const ::InverseJacobian<double, dim, SourceFrame, TargetFrame>&
          inverse_jac,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const override {
    if constexpr ((Maps::supports_hessian && ...)) {
      return inv_hessian_impl(std::move(source_point), inverse_jac, time,
                              functions_of_time);
    } else {
      ERROR("At least one of the Maps does not support autodiff: "
            << CoordinateMap_detail::get_unsupported_autodiff_maps_error<
                   Maps...>());
    }
  }
  InverseHessian<DataVector, dim, SourceFrame, TargetFrame> inv_hessian(
      tnsr::I<DataVector, dim, SourceFrame> source_point,
      const ::InverseJacobian<DataVector, dim, SourceFrame, TargetFrame>&
          inverse_jac,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const override {
    if constexpr ((Maps::supports_hessian && ...)) {
      return inv_hessian_impl(std::move(source_point), inverse_jac, time,
                              functions_of_time);
    } else {
      ERROR("At least one of the Maps does not support autodiff: "
            << CoordinateMap_detail::get_unsupported_autodiff_maps_error<
                   Maps...>());
    }
  }
  /// @}
#endif  // SPECTRE_AUTODIFF

  /// @{
  /// Compute the Jacobian of the `Maps...` at the point(s) `source_point`
  Jacobian<double, dim, SourceFrame, TargetFrame> jacobian(
      tnsr::I<double, dim, SourceFrame> source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const override {
    return jacobian_impl(std::move(source_point), time, functions_of_time);
  }
  Jacobian<DataVector, dim, SourceFrame, TargetFrame> jacobian(
      tnsr::I<DataVector, dim, SourceFrame> source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const override {
    return jacobian_impl(std::move(source_point), time, functions_of_time);
  }
  /// @}

  /// @{
  /// Compute the mapped coordinates, frame velocity, Jacobian, and inverse
  /// Jacobian. The inverse Jacobian is computed by numerically inverting the
  /// Jacobian as this was measured to be quicker than computing it directly for
  /// more complex map concatenations.
  auto coords_frame_velocity_jacobians(
      tnsr::I<double, dim, SourceFrame> source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const
      -> std::tuple<tnsr::I<double, dim, TargetFrame>,
                    InverseJacobian<double, dim, SourceFrame, TargetFrame>,
                    Jacobian<double, dim, SourceFrame, TargetFrame>,
                    tnsr::I<double, dim, TargetFrame>> override {
    return coords_frame_velocity_jacobians_impl(std::move(source_point), time,
                                                functions_of_time);
  }
  auto coords_frame_velocity_jacobians(
      tnsr::I<DataVector, dim, SourceFrame> source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const FunctionsOfTimeMap& functions_of_time = {}) const
      -> std::tuple<tnsr::I<DataVector, dim, TargetFrame>,
                    InverseJacobian<DataVector, dim, SourceFrame, TargetFrame>,
                    Jacobian<DataVector, dim, SourceFrame, TargetFrame>,
                    tnsr::I<DataVector, dim, TargetFrame>> override {
    return coords_frame_velocity_jacobians_impl(std::move(source_point), time,
                                                functions_of_time);
  }
  /// @}

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(CoordinateMapBase<SourceFrame, TargetFrame, dim>),
      CoordinateMap);

  explicit CoordinateMap(CkMigrateMessage* /*unused*/) {}

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    size_t version = 0;
    p | version;
    // Remember to increment the version number when making changes to this
    // function. Retain support for unpacking data written by previous versions
    // whenever possible. See `Domain` docs for details.
    if (version >= 0) {
      CoordinateMapBase<SourceFrame, TargetFrame, dim>::pup(p);
      p | maps_;
    }

    // No need to pup this because it is uniquely determined by the maps
    if (p.isUnpacking()) {
      function_of_time_names_ = CoordinateMap_detail::initialize_names(
          maps_, std::make_index_sequence<sizeof...(Maps)>{});
    }
  }

 private:
  friend bool operator==(const CoordinateMap& lhs, const CoordinateMap& rhs) {
    return lhs.maps_ == rhs.maps_;
  }

  template <typename NewMap, typename LocalSourceFrame,
            typename LocalTargetFrame, typename... LocalMaps, size_t... Is>
  friend CoordinateMap<LocalSourceFrame, LocalTargetFrame, LocalMaps..., NewMap>
  // NOLINTNEXTLINE(readability-redundant-declaration,-warnings-as-errors)
  push_back_impl(
      CoordinateMap<LocalSourceFrame, LocalTargetFrame, LocalMaps...>&& old_map,
      NewMap new_map, std::index_sequence<Is...> /*meta*/);

  template <typename... NewMaps, typename LocalSourceFrame,
            typename LocalTargetFrame, typename... LocalMaps, size_t... Is,
            size_t... Js>
  friend CoordinateMap<LocalSourceFrame, LocalTargetFrame, LocalMaps...,
                       NewMaps...>
  // NOLINTNEXTLINE(readability-redundant-declaration,-warnings-as-errors)
  push_back_impl(
      CoordinateMap<LocalSourceFrame, LocalTargetFrame, LocalMaps...>&& old_map,
      CoordinateMap<LocalSourceFrame, LocalTargetFrame, NewMaps...> new_map,
      std::index_sequence<Is...> /*meta*/, std::index_sequence<Js...> /*meta*/);

  template <typename NewMap, typename LocalSourceFrame,
            typename LocalTargetFrame, typename... LocalMaps, size_t... Is>
  friend CoordinateMap<LocalSourceFrame, LocalTargetFrame, NewMap, LocalMaps...>
  // NOLINTNEXTLINE(readability-redundant-declaration,-warnings-as-errors)
  push_front_impl(
      CoordinateMap<LocalSourceFrame, LocalTargetFrame, LocalMaps...>&& old_map,
      NewMap new_map, std::index_sequence<Is...> /*meta*/);

  template <size_t... Is>
  std::unique_ptr<CoordinateMapBase<SourceFrame, Frame::Grid, dim>>
      get_to_grid_frame_impl(std::index_sequence<Is...> /*meta*/) const;

  bool is_equal_to(const CoordinateMapBase<SourceFrame, TargetFrame, dim>&
                       other) const override {
    const auto& cast_of_other = dynamic_cast<const CoordinateMap&>(other);
    return *this == cast_of_other;
  }

  void check_functions_of_time(
      const FunctionsOfTimeMap& functions_of_time) const;

  template <typename T, size_t... Is>
  tnsr::I<T, dim, TargetFrame> call_impl(
      tnsr::I<T, dim, SourceFrame>&& source_point, double time,
      const FunctionsOfTimeMap& functions_of_time,
      std::index_sequence<Is...> /*meta*/) const;

  template <typename T, size_t... Is>
  std::optional<tnsr::I<T, dim, SourceFrame>> inverse_impl(
      tnsr::I<T, dim, TargetFrame>&& target_point, double time,
      const FunctionsOfTimeMap& functions_of_time,
      std::index_sequence<Is...> /*meta*/) const;

  template <typename T>
  InverseJacobian<T, dim, SourceFrame, TargetFrame> inv_jacobian_impl(
      tnsr::I<T, dim, SourceFrame>&& source_point, double time,
      const FunctionsOfTimeMap& functions_of_time) const;

  template <typename T>
  InverseHessian<T, dim, SourceFrame, TargetFrame> inv_hessian_impl(
      tnsr::I<T, dim, SourceFrame>&& source_point,
      const ::InverseJacobian<T, dim, SourceFrame, TargetFrame>& inverse_jac,
      double time, const FunctionsOfTimeMap& functions_of_time) const;

  template <typename T>
  Jacobian<T, dim, SourceFrame, TargetFrame> jacobian_impl(
      tnsr::I<T, dim, SourceFrame>&& source_point, double time,
      const FunctionsOfTimeMap& functions_of_time) const;

  template <typename T>
  std::tuple<tnsr::I<T, dim, TargetFrame>,
             InverseJacobian<T, dim, SourceFrame, TargetFrame>,
             Jacobian<T, dim, SourceFrame, TargetFrame>,
             tnsr::I<T, dim, TargetFrame>>
  coords_frame_velocity_jacobians_impl(
      tnsr::I<T, dim, SourceFrame> source_point, double time,
      const FunctionsOfTimeMap& functions_of_time) const;

  std::tuple<Maps...> maps_;
  std::unordered_set<std::string> function_of_time_names_;
};

/// \ingroup ComputationalDomainGroup
/// \brief Creates a `CoordinateMap` of `maps...`
template <typename SourceFrame, typename TargetFrame, typename... Maps>
auto make_coordinate_map(Maps&&... maps)
    -> CoordinateMap<SourceFrame, TargetFrame, std::decay_t<Maps>...>;

/// \ingroup ComputationalDomainGroup
/// \brief Creates a `std::unique_ptr<CoordinateMapBase>` of `maps...`
template <typename SourceFrame, typename TargetFrame, typename... Maps>
auto make_coordinate_map_base(Maps&&... maps)
    -> std::unique_ptr<CoordinateMapBase<
        SourceFrame, TargetFrame,
        CoordinateMap<SourceFrame, TargetFrame, std::decay_t<Maps>...>::dim>>;

/// \ingroup ComputationalDomainGroup
/// \brief Creates a `std::vector<std::unique_ptr<CoordinateMapBase>>`
/// containing the result of `make_coordinate_map_base` applied to each
/// argument passed in.
template <typename SourceFrame, typename TargetFrame, typename Arg0,
          typename... Args>
auto make_vector_coordinate_map_base(Arg0&& arg_0, Args&&... remaining_args)
    -> std::vector<std::unique_ptr<
        CoordinateMapBase<SourceFrame, TargetFrame, std::decay_t<Arg0>::dim>>>;

/// \ingroup ComputationalDomainGroup
/// \brief Creates a `std::vector<std::unique_ptr<CoordinateMapBase>>`
/// containing the result of `make_coordinate_map_base` applied to each
/// element of the vector of maps composed with the rest of the arguments
/// passed in.
template <typename SourceFrame, typename TargetFrame, size_t Dim, typename Map,
          typename... Maps>
auto make_vector_coordinate_map_base(std::vector<Map> maps,
                                     const Maps&... remaining_maps)
    -> std::vector<
        std::unique_ptr<CoordinateMapBase<SourceFrame, TargetFrame, Dim>>>;

/// \ingroup ComputationalDomainGroup
/// \brief Creates a `CoordinateMap` by appending the new map to the end of the
/// old maps
template <typename SourceFrame, typename TargetFrame, typename... Maps,
          typename NewMap>
CoordinateMap<SourceFrame, TargetFrame, Maps..., NewMap> push_back(
    CoordinateMap<SourceFrame, TargetFrame, Maps...> old_map, NewMap new_map);

/// \ingroup ComputationalDomainGroup
/// \brief Creates a `CoordinateMap` by prepending the new map to the beginning
/// of the old maps
template <typename SourceFrame, typename TargetFrame, typename... Maps,
          typename NewMap>
CoordinateMap<SourceFrame, TargetFrame, NewMap, Maps...> push_front(
    CoordinateMap<SourceFrame, TargetFrame, Maps...> old_map, NewMap new_map);

/// \cond
template <typename SourceFrame, typename TargetFrame, typename... Maps>
PUP::able::PUP_ID
    CoordinateMap<SourceFrame, TargetFrame, Maps...>::my_PUP_ID =  // NOLINT
    0;
/// \endcond
}  // namespace domain
