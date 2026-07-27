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
  using ShapeMapOptionType = typename ShapeMapOptions::type::value_type;

  using RotationMapOptions = time_dependent_options::RotationMapOptions<true>;
  using RotationMapOptionType = typename RotationMapOptions::type::value_type;

  using ExpansionMapOptions = time_dependent_options::ExpansionMapOptions<true>;
  using ExpansionMapOptionType = typename ExpansionMapOptions::type::value_type;

  using TranslationMapOptions =
      time_dependent_options::TranslationMapOptions<3>;
  using TranslationMapOptionType =
      typename TranslationMapOptions::type::value_type;

  struct TransitionRotScaleTrans {
    using type = bool;
    static constexpr Options::String help = {
        "Transition rotation, expansion, and translation to zero in the outer "
        "shell"};
  };

  struct NumberOfRadialShellsWithShapeMap {
    using type = Options::Auto<size_t>;
    static constexpr Options::String help = {
        "Number of innermost radial shells that use the shape map. This must "
        "be at least one. A filled Sphere with multiple radial shells requires "
        "at least two shells with a shape map. For any domain with multiple "
        "radial shells, at least one outer shell must remain without a shape "
        "map. Specify 'Auto' to use the default for the domain."};
  };

  using options =
      tmpl::list<InitialTime, ShapeMapOptions, RotationMapOptions,
                 ExpansionMapOptions, TranslationMapOptions,
                 TransitionRotScaleTrans, NumberOfRadialShellsWithShapeMap>;
  static constexpr Options::String help{
      "The options for all the hard-coded time dependent maps in the "
      "Sphere domain."};

  TimeDependentMapOptions() = default;

  TimeDependentMapOptions(
      double initial_time, ShapeMapOptionType shape_map_options,
      RotationMapOptionType rotation_map_options,
      ExpansionMapOptionType expansion_map_options,
      TranslationMapOptionType translation_map_options,
      bool transition_rot_scale_trans,
      std::optional<size_t> number_of_radial_shells_with_shape_map);

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
  create_functions_of_time(const std::unordered_map<std::string, double>&
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
  void build_maps(const std::array<double, 3>& center, bool is_filled,
                  double inner_radius,
                  const std::vector<double>& radial_partitions,
                  double outer_radius);

  /*!
   * \brief This will construct the map from `Frame::Distorted` to
   * `Frame::Inertial`.
   *
   * For blocks with a shape map, this will be a RotScaleTrans map. For other
   * blocks, this returns `nullptr`.
   *
   * \param radial_shell Zero-based index specifying a shell in the `Sphere`
   * or `SphericalShells` domain, starting at the innermost shell.
   * \param is_inner_cube Whether the block is the central cube of a filled
   * sphere.
   */
  MapType<Frame::Distorted, Frame::Inertial> distorted_to_inertial_map(
      size_t radial_shell, bool is_inner_cube) const;

  /*!
   * \brief This will construct the map from `Frame::Grid` to
   * `Frame::Distorted`.
   *
   * For blocks with a shape map, this will return the `Shape` map (with a size
   * function of time). For other blocks, this returns `nullptr`.
   *
   * \param radial_shell Zero-based index specifying a shell in the `Sphere`
   * or `SphericalShells` domain, starting at the innermost shell.
   * \param shape_map_index Index of the wedge direction in the order returned
   * by `orientations_for_sphere_wrappings`. This is only used for filled
   * spheres. In that case, the implementation uses the Wedge transition
   * function and builds two sets of shape maps for the six canonical wedge
   * directions: indices 0..5 deform the innermost shell and indices 6..11 roll
   * off the deformation in outer shells. If the sphere is not filled, then
   * the implementation uses the SphereTransition function and builds only one
   * shape map, leaving this option unused.
   * \param is_inner_cube Whether the block is the central cube of a filled
   * sphere.
   */
  MapType<Frame::Grid, Frame::Distorted> grid_to_distorted_map(
      size_t radial_shell, size_t shape_map_index, bool is_inner_cube) const;

  /*!
   * \brief This will construct the map from `Frame::Grid` to `Frame::Inertial`.
   *
   * For blocks with a shape map, this will return the `Shape` and
   * `RotScaleTrans` composition. For other blocks, this returns just the
   * `RotScaleTrans` map. In the outer shell, the `RotScaleTrans` map will
   * transition to zero.
   *
   * \param radial_shell Zero-based index specifying a shell in the `Sphere`
   * or `SphericalShells` domain, starting at the innermost shell.
   * \param shape_map_index Index of the wedge direction in the order returned
   * by `orientations_for_sphere_wrappings`. This is only used for filled
   * spheres. In that case, the implementation uses the Wedge transition
   * function and builds two sets of shape maps for the six canonical wedge
   * directions: indices 0..5 deform the innermost shell and indices 6..11 roll
   * off the deformation in outer shells. If the sphere is not filled, then
   * the implementation uses the SphereTransition function and builds only one
   * shape map, leaving this option unused.
   * \param is_outer_shell Whether the block is in the outermost radial shell.
   * \param is_central_region Whether the map is for the central cube of a
   * filled sphere or the excision boundary of an excised sphere.
   */
  MapType<Frame::Grid, Frame::Inertial> grid_to_inertial_map(
      size_t radial_shell, size_t shape_map_index, bool is_outer_shell,
      bool is_central_region) const;

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
  bool filled_{false};
  double deformed_radius_{std::numeric_limits<double>::signaling_NaN()};
  std::array<ShapeMap, 12> shape_maps_{};
  RotScaleTransMap inner_rot_scale_trans_map_{};
  RotScaleTransMap transition_rot_scale_trans_map_{};

  ShapeMapOptionType shape_map_options_;
  RotationMapOptionType rotation_map_options_;
  ExpansionMapOptionType expansion_map_options_;
  TranslationMapOptionType translation_map_options_;
  bool transition_rot_scale_trans_{false};
  std::optional<size_t> number_of_radial_shells_with_shape_map_{};
};
}  // namespace domain::creators::sphere
