// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/TimeDependent/RotScaleTrans.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Shape.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/TimeDependentOptions/ExpansionMap.hpp"
#include "Domain/Creators/TimeDependentOptions/RotationMap.hpp"
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/Creators/TimeDependentOptions/TranslationMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Grid;
struct Distorted;
struct Inertial;
}  // namespace Frame
/// \endcond

namespace domain::creators::sphere {
/*!
 * \brief This holds all options related to the time dependent maps of the
 * domain::creators::Sphere domain creator.
 *
 * \details Currently this class will only add a Shape map (and size
 * FunctionOfTime) to the domain. Other maps can be added as needed.
 *
 * \note This struct contains no information about what blocks the time
 * dependent maps will go in.
 */
struct TimeDependentMapOptions {
 private:
  template <typename SourceFrame, typename TargetFrame>
  using MapType =
      std::unique_ptr<domain::CoordinateMapBase<SourceFrame, TargetFrame, 3>>;
  using IdentityMap = domain::CoordinateMaps::Identity<3>;
  // Time-dependent maps
  using ShapeMap = domain::CoordinateMaps::TimeDependent::Shape;
  using RotScaleTransMap =
      domain::CoordinateMaps::TimeDependent::RotScaleTrans<3>;

  template <typename SourceFrame, typename TargetFrame>
  using IdentityForComposition =
      domain::CoordinateMap<SourceFrame, TargetFrame, IdentityMap>;
  using GridToDistortedComposition =
      domain::CoordinateMap<Frame::Grid, Frame::Distorted, ShapeMap>;
  using GridToInertialComposition =
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, ShapeMap,
                            RotScaleTransMap>;
  using GridToInertialSimple =
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, RotScaleTransMap>;
  using DistortedToInertialComposition =
      domain::CoordinateMap<Frame::Distorted, Frame::Inertial,
                            RotScaleTransMap>;
  using GridToInertialShapeMap =
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, ShapeMap>;

 public:
  using maps_list =
      tmpl::list<IdentityForComposition<Frame::Grid, Frame::Inertial>,
                 IdentityForComposition<Frame::Grid, Frame::Distorted>,
                 IdentityForComposition<Frame::Distorted, Frame::Inertial>,
                 GridToDistortedComposition, GridToInertialShapeMap,
                 GridToInertialSimple, GridToInertialComposition,
                 DistortedToInertialComposition>;

  /// \brief The initial time of the functions of time.
  struct InitialTime {
    using type = double;
    static constexpr Options::String help = {
        "The initial time of the functions of time"};
  };

  using ShapeMapOptions =
      time_dependent_options::ShapeMapOptions<false, domain::ObjectLabel::None>;

  using RotationMapOptions = time_dependent_options::RotationMapOptions<2>;

  using ExpansionMapOptions = time_dependent_options::ExpansionMapOptions;

  using TranslationMapOptions =
      time_dependent_options::TranslationMapOptions<3>;

  using options = tmpl::list<InitialTime, ShapeMapOptions, RotationMapOptions,
                             ExpansionMapOptions, TranslationMapOptions>;
  static constexpr Options::String help{
      "The options for all the hard-coded time dependent maps in the "
      "Sphere domain."};

  TimeDependentMapOptions() = default;

  TimeDependentMapOptions(
      double initial_time, std::optional<ShapeMapOptions> shape_map_options,
      std::optional<RotationMapOptions> rotation_map_options,
      std::optional<ExpansionMapOptions> expansion_map_options,
      std::optional<TranslationMapOptions> translation_map_options);

  /*!
   * \brief Create the function of time map using the options that were
   * provided to this class.
   *
   * Currently, this will add:
   *
   * - Size: `PiecewisePolynomial<3>`
   * - Shape: `PiecewisePolynomial<2>`
   * - Rotation: `SettleToConstantQuaternion`
   * - Expansion: `SettleToConstant`
   * - ExpansionOuterBoundary: `PiecewisePolynomial<2>`
   * - Translation: `PiecewisePolynomial<2>`
   */
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
  create_functions_of_time(double inner_radius,
                           const std::unordered_map<std::string, double>&
                               initial_expiration_times) const;

  /*!
   * \brief Construct the actual maps that will be used.
   *
   * Currently, this constructs a:
   *
   * - Shape: `Shape` (with a size function of time)
   * - Rotation: `Rotation`
   * - Expansion: `Expansion`
   * - Expansion outside the transition region: `ExpansionOuterBoundary`
   * - Translation: `Translation`
   */
  void build_maps(const std::array<double, 3>& center,
                  std::pair<double, double> inner_shell_radii,
                  std::pair<double, double> outer_shell_radii);

  /*!
   * \brief This will construct the map from `Frame::Distorted` to
   * `Frame::Inertial`.
   *
   * If the argument `include_distorted_map` is true, then this will be a
   * RotScaleTrans map. If it is false, then this returns `nullptr`.
   */
  MapType<Frame::Distorted, Frame::Inertial> distorted_to_inertial_map(
      bool include_distorted_map) const;

  /*!
   * \brief This will construct the map from `Frame::Grid` to
   * `Frame::Distorted`.
   *
   * If the argument `include_distorted_map` is true, then this will add a
   * `Shape` map (with a size function of time). If it is false, then this
   * returns `nullptr`.
   */
  MapType<Frame::Grid, Frame::Distorted> grid_to_distorted_map(
      bool include_distorted_map) const;

  /*!
   * \brief This will construct the map from `Frame::Grid` to
   * `Frame::Inertial`.
   *
   * If the argument `include_distorted_map` is true, then this map will
   * have a `Shape` map (with a size function of time). If it is false, then
   * there will be a RotScaleTrans map in either the inner region or in the
   * transition region.
   */
  MapType<Frame::Grid, Frame::Inertial> grid_to_inertial_map(
      bool include_distorted_map, bool use_rigid) const;

  /*!
   * \brief Whether or not the distorted frame is being used. I.e. whether or
   * not shape map options were specified.
   */
  bool using_distorted_frame() const;

  inline static const std::string size_name{"Size"};
  inline static const std::string shape_name{"Shape"};
  inline static const std::string rotation_name{"Rotation"};
  inline static const std::string expansion_name{"Expansion"};
  inline static const std::string expansion_outer_boundary_name{
      "ExpansionOuterBoundary"};
  inline static const std::string translation_name{"Translation"};

 private:
  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
  std::optional<ShapeMap> shape_map_{};
  RotScaleTransMap inner_rot_scale_trans_map_{};
  RotScaleTransMap transition_rot_scale_trans_map_{};

  std::optional<ShapeMapOptions> shape_map_options_{};
  std::optional<RotationMapOptions> rotation_map_options_{};
  std::optional<ExpansionMapOptions> expansion_map_options_{};
  std::optional<TranslationMapOptions> translation_map_options_{};
};
}  // namespace domain::creators::sphere
