// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class CoordinateMap

#pragma once

#include <boost/optional.hpp>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
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

  /// Returns `true` if the map is the identity
  virtual bool is_identity() const noexcept = 0;

  // @{
  /// Apply the `Maps` to the point(s) `source_point`
  virtual tnsr::I<double, Dim, TargetFrame> operator()(
      tnsr::I<double, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time = std::unordered_map<
              std::string,
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{}) const
      noexcept = 0;
  virtual tnsr::I<DataVector, Dim, TargetFrame> operator()(
      tnsr::I<DataVector, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time = std::unordered_map<
              std::string,
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{}) const
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
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time = std::unordered_map<
              std::string,
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{}) const
      noexcept = 0;
  // @}

  // @{
  /// Compute the inverse Jacobian of the `Maps` at the point(s)
  /// `source_point`
  virtual InverseJacobian<double, Dim, SourceFrame, TargetFrame> inv_jacobian(
      tnsr::I<double, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time = std::unordered_map<
              std::string,
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{}) const
      noexcept = 0;
  virtual InverseJacobian<DataVector, Dim, SourceFrame, TargetFrame>
  inv_jacobian(
      tnsr::I<DataVector, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time = std::unordered_map<
              std::string,
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{}) const
      noexcept = 0;
  // @}

  // @{
  /// Compute the Jacobian of the `Maps` at the point(s) `source_point`
  virtual Jacobian<double, Dim, SourceFrame, TargetFrame> jacobian(
      tnsr::I<double, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time = std::unordered_map<
              std::string,
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{}) const
      noexcept = 0;
  virtual Jacobian<DataVector, Dim, SourceFrame, TargetFrame> jacobian(
      tnsr::I<DataVector, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time = std::unordered_map<
              std::string,
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{}) const
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

  explicit CoordinateMap(Maps... maps);

  std::unique_ptr<CoordinateMapBase<SourceFrame, TargetFrame, dim>> get_clone()
      const override {
    return std::make_unique<CoordinateMap>(*this);
  }

  /// Returns `true` if the map is the identity
  bool is_identity() const noexcept override;

  // @{
  /// Apply the `Maps...` to the point(s) `source_point`
  constexpr tnsr::I<double, dim, TargetFrame> operator()(
      tnsr::I<double, dim, SourceFrame> source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time = std::unordered_map<
              std::string,
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{}) const
      noexcept override {
    return call_impl(std::move(source_point), time, functions_of_time,
                     std::make_index_sequence<sizeof...(Maps)>{});
  }
  constexpr tnsr::I<DataVector, dim, TargetFrame> operator()(
      tnsr::I<DataVector, dim, SourceFrame> source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time = std::unordered_map<
              std::string,
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{}) const
      noexcept override {
    return call_impl(std::move(source_point), time, functions_of_time,
                     std::make_index_sequence<sizeof...(Maps)>{});
  }
  // @}

  // @{
  /// Apply the inverse `Maps...` to the point(s) `target_point`
  constexpr boost::optional<tnsr::I<double, dim, SourceFrame>> inverse(
      tnsr::I<double, dim, TargetFrame> target_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time = std::unordered_map<
              std::string,
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{}) const
      noexcept override {
    return inverse_impl(std::move(target_point), time, functions_of_time,
                        std::make_index_sequence<sizeof...(Maps)>{});
  }
  // @}

  // @{
  /// Compute the inverse Jacobian of the `Maps...` at the point(s)
  /// `source_point`
  constexpr InverseJacobian<double, dim, SourceFrame, TargetFrame> inv_jacobian(
      tnsr::I<double, dim, SourceFrame> source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time = std::unordered_map<
              std::string,
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{}) const
      noexcept override {
    return inv_jacobian_impl(std::move(source_point), time, functions_of_time);
  }
  constexpr InverseJacobian<DataVector, dim, SourceFrame, TargetFrame>
  inv_jacobian(
      tnsr::I<DataVector, dim, SourceFrame> source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time = std::unordered_map<
              std::string,
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{}) const
      noexcept override {
    return inv_jacobian_impl(std::move(source_point), time, functions_of_time);
  }
  // @}

  // @{
  /// Compute the Jacobian of the `Maps...` at the point(s) `source_point`
  constexpr Jacobian<double, dim, SourceFrame, TargetFrame> jacobian(
      tnsr::I<double, dim, SourceFrame> source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time = std::unordered_map<
              std::string,
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{}) const
      noexcept override {
    return jacobian_impl(std::move(source_point), time, functions_of_time);
  }
  constexpr Jacobian<DataVector, dim, SourceFrame, TargetFrame> jacobian(
      tnsr::I<DataVector, dim, SourceFrame> source_point,
      const double time = std::numeric_limits<double>::signaling_NaN(),
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time = std::unordered_map<
              std::string,
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{}) const
      noexcept override {
    return jacobian_impl(std::move(source_point), time, functions_of_time);
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

  template <typename NewMap, typename LocalSourceFrame,
            typename LocalTargetFrame, typename... LocalMaps, size_t... Is>
  friend CoordinateMap<LocalSourceFrame, LocalTargetFrame, LocalMaps..., NewMap>
  // NOLINTNEXTLINE(readability-redundant-declaration,-warnings-as-errors)
  push_back_impl(
      CoordinateMap<LocalSourceFrame, LocalTargetFrame, LocalMaps...>&& old_map,
      NewMap new_map, std::index_sequence<Is...> /*meta*/) noexcept;

  template <typename... NewMaps, typename LocalSourceFrame,
            typename LocalTargetFrame, typename... LocalMaps, size_t... Is,
            size_t... Js>
  friend CoordinateMap<LocalSourceFrame, LocalTargetFrame, LocalMaps...,
                       NewMaps...>
  // NOLINTNEXTLINE(readability-redundant-declaration,-warnings-as-errors)
  push_back_impl(
      CoordinateMap<LocalSourceFrame, LocalTargetFrame, LocalMaps...>&& old_map,
      CoordinateMap<LocalSourceFrame, LocalTargetFrame, NewMaps...> new_map,
      std::index_sequence<Is...> /*meta*/,
      std::index_sequence<Js...> /*meta*/) noexcept;

  template <typename NewMap, typename LocalSourceFrame,
            typename LocalTargetFrame, typename... LocalMaps, size_t... Is>
  friend CoordinateMap<LocalSourceFrame, LocalTargetFrame, NewMap, LocalMaps...>
  // NOLINTNEXTLINE(readability-redundant-declaration,-warnings-as-errors)
  push_front_impl(
      CoordinateMap<LocalSourceFrame, LocalTargetFrame, LocalMaps...>&& old_map,
      NewMap new_map, std::index_sequence<Is...> /*meta*/) noexcept;

  bool is_equal_to(const CoordinateMapBase<SourceFrame, TargetFrame, dim>&
                       other) const override {
    const auto& cast_of_other = dynamic_cast<const CoordinateMap&>(other);
    return *this == cast_of_other;
  }

  template <typename T, size_t... Is>
  tnsr::I<T, dim, TargetFrame> call_impl(
      tnsr::I<T, dim, SourceFrame>&& source_point, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      std::index_sequence<Is...> /*meta*/) const noexcept;

  template <typename T, size_t... Is>
  boost::optional<tnsr::I<T, dim, SourceFrame>> inverse_impl(
      tnsr::I<T, dim, TargetFrame>&& target_point, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      std::index_sequence<Is...> /*meta*/) const noexcept;

  template <typename T>
  InverseJacobian<T, dim, SourceFrame, TargetFrame> inv_jacobian_impl(
      tnsr::I<T, dim, SourceFrame>&& source_point, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;

  template <typename T>
  Jacobian<T, dim, SourceFrame, TargetFrame> jacobian_impl(
      tnsr::I<T, dim, SourceFrame>&& source_point, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;

  std::tuple<Maps...> maps_;
};

/// \ingroup ComputationalDomainGroup
/// \brief Creates a `CoordinateMap` of `maps...`
template <typename SourceFrame, typename TargetFrame, typename... Maps>
auto make_coordinate_map(Maps&&... maps) noexcept
    -> CoordinateMap<SourceFrame, TargetFrame, std::decay_t<Maps>...>;

/// \ingroup ComputationalDomainGroup
/// \brief Creates a `std::unique_ptr<CoordinateMapBase>` of `maps...`
template <typename SourceFrame, typename TargetFrame, typename... Maps>
auto make_coordinate_map_base(Maps&&... maps) noexcept
    -> std::unique_ptr<CoordinateMapBase<
        SourceFrame, TargetFrame,
        CoordinateMap<SourceFrame, TargetFrame, std::decay_t<Maps>...>::dim>>;

/// \ingroup ComputationalDomainGroup
/// \brief Creates a `std::vector<std::unique_ptr<CoordinateMapBase>>`
/// containing the result of `make_coordinate_map_base` applied to each
/// argument passed in.
template <typename SourceFrame, typename TargetFrame, typename Arg0,
          typename... Args>
auto make_vector_coordinate_map_base(Arg0&& arg_0,
                                     Args&&... remaining_args) noexcept
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
                                     const Maps&... remaining_maps) noexcept
    -> std::vector<
        std::unique_ptr<CoordinateMapBase<SourceFrame, TargetFrame, Dim>>>;

/// \ingroup ComputationalDomainGroup
/// \brief Creates a `CoordinateMap` by appending the new map to the end of the
/// old maps
template <typename SourceFrame, typename TargetFrame, typename... Maps,
          typename NewMap>
CoordinateMap<SourceFrame, TargetFrame, Maps..., NewMap> push_back(
    CoordinateMap<SourceFrame, TargetFrame, Maps...> old_map,
    NewMap new_map) noexcept;

/// \ingroup ComputationalDomainGroup
/// \brief Creates a `CoordinateMap` by prepending the new map to the beginning
/// of the old maps
template <typename SourceFrame, typename TargetFrame, typename... Maps,
          typename NewMap>
CoordinateMap<SourceFrame, TargetFrame, NewMap, Maps...> push_front(
    CoordinateMap<SourceFrame, TargetFrame, Maps...> old_map,
    NewMap new_map) noexcept;

/// \cond
template <typename SourceFrame, typename TargetFrame, typename... Maps>
PUP::able::PUP_ID
    CoordinateMap<SourceFrame, TargetFrame, Maps...>::my_PUP_ID =  // NOLINT
    0;
/// \endcond
}  // namespace domain
