// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/CoordinateMaps/TimeDependent/RotScaleTrans.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Rotation.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Shape.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/Creators/TimeDependentOptions/Sphere.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/Options.hpp"
#include "Options/String.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Grid;
struct Distorted;
struct Inertial;
}  // namespace Frame
/// \endcond

/// Namespace used to hold things used in both the BinaryCompactObject and
/// CylindricalBinaryCompactObject domain creators.
namespace domain::creators::bco {
namespace detail {
// Convenience type alias
template <typename... Maps>
using gi_map = domain::CoordinateMap<Frame::Grid, Frame::Inertial, Maps...>;
template <typename... Maps>
using gd_map = domain::CoordinateMap<Frame::Grid, Frame::Distorted, Maps...>;
template <typename... Maps>
using di_map =
    domain::CoordinateMap<Frame::Distorted, Frame::Inertial, Maps...>;

template <typename List>
struct power_set {
  using rest = typename power_set<tmpl::pop_front<List>>::type;
  using type = tmpl::append<
      rest, tmpl::transform<rest, tmpl::lazy::push_front<
                                      tmpl::_1, tmpl::pin<tmpl::front<List>>>>>;
};

template <>
struct power_set<tmpl::list<>> {
  using type = tmpl::list<tmpl::list<>>;
};

template <typename SourceFrame, typename TargetFrame, typename Maps>
using produce_all_maps_helper =
    tmpl::wrap<tmpl::push_front<Maps, SourceFrame, TargetFrame>,
               domain::CoordinateMap>;

/*
 * This will produce all the possible combinations of maps we will need for the
 * maps_list. It does so using a power set. Say you have 3 maps, Map1, Map2,
 * Map3. The end result of this will be a list with the following map
 * combinations where these combinations may not appear in the same position in
 * the list, but the order of each combination is fixed:
 *  - Map1
 *  - Map2
 *  - Map3
 *  - Map1, Map2
 *  - Map1, Map3
 *  - Map2, Map3
 *  - Map1, Map2, Map3
 */
template <typename SourceFrame, typename TargetFrame, typename... Maps>
using produce_all_maps = tmpl::transform<
    tmpl::remove<typename power_set<tmpl::list<Maps...>>::type, tmpl::list<>>,
    tmpl::bind<produce_all_maps_helper, tmpl::pin<SourceFrame>,
               tmpl::pin<TargetFrame>, tmpl::_1>>;
}  // namespace detail

/*!
 * \brief This holds all options related to the time dependent maps of the
 * binary compact object domains.
 *
 * \details Since both domains will have the same (overall) time dependent maps,
 * their options are going to be the same as well. The options won't be exactly
 * the same though, so there is a \p IsCylindrical template parameter to
 * distinguish.
 *
 * This class will create the FunctionsOfTime needed for the binary compact
 * object domains as well as the actual CoordinateMap%s themselves
 *
 * \note This struct contains no information about what blocks the time
 * dependent maps will go in.
 */
template <bool IsCylindrical>
struct TimeDependentMapOptions {
 private:
  template <typename SourceFrame, typename TargetFrame>
  using MapType =
      std::unique_ptr<domain::CoordinateMapBase<SourceFrame, TargetFrame, 3>>;
  // Time-dependent maps
  using Expansion = domain::CoordinateMaps::TimeDependent::CubicScale<3>;
  using Rotation = domain::CoordinateMaps::TimeDependent::Rotation<3>;
  using RotScaleTrans = domain::CoordinateMaps::TimeDependent::RotScaleTrans<3>;
  using Shape = domain::CoordinateMaps::TimeDependent::Shape;
  using Identity = domain::CoordinateMaps::Identity<3>;

 public:
  using maps_list = tmpl::append<
      // We need this odd-one-out Identity map because all maps are optional to
      // specify. It's possible to specify a shape map, but no expansion or
      // rotation map. So we have a grid to distorted map, and need an identity
      // distorted to inertial map. We don't need an identity grid to distorted
      // map because if a user requests the grid to distorted frame map with a
      // distorted frame, but didn't specify shape map options, an error occurs.
      tmpl::list<detail::di_map<Identity>>,
      detail::produce_all_maps<Frame::Grid, Frame::Inertial, Shape, Expansion,
                               Rotation>,
      detail::produce_all_maps<Frame::Grid, Frame::Distorted, Shape>,
      detail::produce_all_maps<Frame::Distorted, Frame::Inertial, Expansion,
                               Rotation>,
      detail::produce_all_maps<Frame::Grid, Frame::Inertial, Shape,
                               RotScaleTrans>,
      detail::produce_all_maps<Frame::Distorted, Frame::Inertial,
                               RotScaleTrans>>;

  /// \brief The initial time of the functions of time.
  struct InitialTime {
    using type = double;
    static constexpr Options::String help = {
        "The initial time of the functions of time"};
  };

  /// \brief Options for the expansion map.
  /// The outer boundary radius of the map is always set to
  /// the outer boundary of the Domain, so there is no option
  /// here to set the outer boundary radius.
  struct ExpansionMapOptions {
    using type = Options::Auto<ExpansionMapOptions, Options::AutoLabel::None>;
    static std::string name() { return "ExpansionMap"; }
    static constexpr Options::String help = {
        "Options for the expansion map. Specify 'None' to not use this map."};
    struct InitialValues {
      using type = std::array<double, 2>;
      static constexpr Options::String help = {
          "Initial value and deriv of expansion."};
    };
    struct AsymptoticVelocityOuterBoundary {
      using type = double;
      static constexpr Options::String help = {
          "The asymptotic velocity of the outer boundary."};
    };
    struct DecayTimescaleOuterBoundaryVelocity {
      using type = double;
      static constexpr Options::String help = {
          "The timescale for how fast the outer boundary velocity approaches "
          "its asymptotic value."};
    };
    using options = tmpl::list<InitialValues, AsymptoticVelocityOuterBoundary,
                               DecayTimescaleOuterBoundaryVelocity>;
    ExpansionMapOptions() = default;
    ExpansionMapOptions(std::array<double, 2> initial_values_in,
                        double outer_boundary_velocity_in,
                        double outer_boundary_decay_time_in)
        : initial_values(initial_values_in),
          outer_boundary_velocity(outer_boundary_velocity_in),
          outer_boundary_decay_time(outer_boundary_decay_time_in) {}

    std::array<double, 2> initial_values{
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN()};
    double outer_boundary_velocity{
        std::numeric_limits<double>::signaling_NaN()};
    double outer_boundary_decay_time{
        std::numeric_limits<double>::signaling_NaN()};
  };

  struct RotationMapOptions {
    using type = Options::Auto<RotationMapOptions, Options::AutoLabel::None>;
    static std::string name() { return "RotationMap"; }
    static constexpr Options::String help = {
        "Options for a time-dependent rotation map about an arbitrary axis. "
        "Specify 'None' to not use this map."};

    struct InitialAngularVelocity {
      using type = std::array<double, 3>;
      static constexpr Options::String help = {"The initial angular velocity."};
    };

    using options = tmpl::list<InitialAngularVelocity>;

    RotationMapOptions() = default;
    explicit RotationMapOptions(
        std::array<double, 3> initial_angular_velocity_in)
        : initial_angular_velocity(initial_angular_velocity_in) {}

    std::array<double, 3> initial_angular_velocity{};
  };

  /// \brief Options for the Translation Map, the outer radius is always set to
  /// the outer boundary of the Domain, so there's no option needed for outer
  /// boundary.
  struct TranslationMapOptions {
    using type = Options::Auto<TranslationMapOptions, Options::AutoLabel::None>;
    static std::string name() { return "TranslationMap"; }
    static constexpr Options::String help = {
        "Options for a time-dependent translation map. Specify 'None' to not "
        "use this map."};

    struct InitialValues {
      using type = std::array<std::array<double, 3>, 3>;
      static constexpr Options::String help = {
          "Initial position, velocity and acceleration."};
    };

    using options = tmpl::list<InitialValues>;
    TranslationMapOptions() = default;
    explicit TranslationMapOptions(
        std::array<std::array<double, 3>, 3> initial_values_in)
        : initial_values(initial_values_in) {}

    std::array<std::array<double, 3>, 3> initial_values{};
  };

  // We use a type alias here instead of defining the ShapeMapOptions struct
  // because there appears to be a bug in clang-10. If the definition of
  // ShapeMapOptions is here inside TimeDependentMapOptions, on clang-10 there
  // is a linking error that there is an undefined reference to
  // Options::Option::parse_as<TimeDependentMapOptions<A>> (and B). This doesn't
  // show up for GCC. If we put the definition of ShapeMapOptions outside of
  // TimeDependentMapOptions and just use a type alias here, the linking error
  // goes away.
  template <domain::ObjectLabel Object>
  using ShapeMapOptions =
      domain::creators::time_dependent_options::ShapeMapOptions<
          not IsCylindrical, Object>;

  using options =
      tmpl::list<InitialTime, ExpansionMapOptions, RotationMapOptions,
                 TranslationMapOptions, ShapeMapOptions<domain::ObjectLabel::A>,
                 ShapeMapOptions<domain::ObjectLabel::B>>;
  static constexpr Options::String help{
      "The options for all time dependent maps in a binary compact object "
      "domain. Specify 'None' to not use any time dependent maps."};

  TimeDependentMapOptions() = default;

  TimeDependentMapOptions(
      double initial_time,
      std::optional<ExpansionMapOptions> expansion_map_options,
      std::optional<RotationMapOptions> rotation_map_options,
      std::optional<TranslationMapOptions> translation_map_options,
      std::optional<ShapeMapOptions<domain::ObjectLabel::A>> shape_options_A,
      std::optional<ShapeMapOptions<domain::ObjectLabel::B>> shape_options_B,
      const Options::Context& context = {});

  /*!
   * \brief Create the function of time map using the options that were
   * provided to this class.
   *
   * Currently, this will add:
   *
   * - Expansion: `PiecewisePolynomial<2>`
   * - ExpansionOuterBoundary: `FixedSpeedCubic`
   * - Rotation: `QuaternionFunctionOfTime<3>`
   * - Translation: `PiecewisePolynomial<3>`
   * - SizeA/B: `PiecewisePolynomial<3>`
   * - ShapeA/B: `PiecewisePolynomial<2>`
   *
   *  When `UseWorldtube` is set to true, they are
   *
   * - Expansion: `IntegratedFunctionOfTime`
   * - ExpansionOuterBoundary: `FixedSpeedCubic`
   * - Rotation: `IntegratedFunctionOfTime`
   * - Translation: None
   * - SizeA/B: IntegratedFunctionOfTime
   * - ShapeA/B: `PiecewisePolynomial<2>`
   */
  template <bool UseWorldtube = false>
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
  create_functions_of_time(const std::unordered_map<std::string, double>&
                               initial_expiration_times) const;

  /*!
   * \brief Construct the actual maps that will be used.
   *
   * Currently, this constructs a:
   *
   * - Rotation, Expansion, Translation: `RotScaleTrans<3>`
   * - ShapeA/B: `Shape` (with size FunctionOfTime)
   *
   * If the radii for an object are `std::nullopt`, this means that a Shape map
   * is not constructed for that object. An identity map will be used instead.
   * If \p IsCylindrical is true, only pass two radii for the inner/outer radius
   * of the object sphere. If it is false, pass three radii corresponding to the
   * excision radius, the outer radius of the inner sphere, and the radius of
   * the surrounding cube.
   */
  void build_maps(
      const std::array<std::array<double, 3>, 2>& object_centers,
      const std::optional<std::array<double, 3>>& cube_A_center,
      const std::optional<std::array<double, 3>>& cube_B_center,
      const std::optional<std::array<double, IsCylindrical ? 2 : 3>>&
          object_A_radii,
      const std::optional<std::array<double, IsCylindrical ? 2 : 3>>&
          object_B_radii,
      double envelope_radius, double domain_outer_radius);

  /*!
   * \brief Check whether options were specified in the constructor for the
   * shape map of this object
   */
  bool has_distorted_frame_options(domain::ObjectLabel object) const;

  /*!
   * \brief Type to pass to `grid_to_distorted_map()` and
   * `grid_to_inertial_map()`.
   *
   * \details If \p IsCylindrical is true, pass a `bool` for whether to include
   * a shape map or not. If it's false, pass a `std::optional<size_t>`. If this
   * has a value, then it includes a shape map. The `size_t` represents the
   * relative block number around each object in the BinaryCompactObject domain.
   * It should go from 0 to 11 for the 12 blocks surrounding each object.
   */
  using IncludeDistortedMapType =
      tmpl::conditional_t<IsCylindrical, bool, std::optional<size_t>>;

  /*!
   * \brief This will construct the map from `Frame::Distorted` to
   * `Frame::Inertial`
   *
   * If we are including a shape map, then this will be a `RotScaleTrans` map.
   * If we aren't, this returns a `nullptr`.
   *
   * \see IncludeDistortedMapType
   */
  template <domain::ObjectLabel Object>
  MapType<Frame::Distorted, Frame::Inertial> distorted_to_inertial_map(
      const IncludeDistortedMapType& include_distorted_map,
      bool use_rigid_map) const;

  /*!
   * \brief This will construct the maps from the `Frame::Grid` to the
   * `Frame::Distorted`.
   *
   * If we are including a shape map, then this will be a `Shape` map (with size
   * FunctionOfTime) for the templated `Object`. If we aren't, then this returns
   * a `nullptr`.
   *
   * \see IncludeDistortedMapType
   */
  template <domain::ObjectLabel Object>
  MapType<Frame::Grid, Frame::Distorted> grid_to_distorted_map(
      const IncludeDistortedMapType& include_distorted_map) const;

  /*!
   * \brief This will construct the entire map from the `Frame::Grid` to the
   * `Frame::Inertial`.
   *
   * If we are including a shape map, then this map will have a composition of a
   * `Shape` (with size FunctionOfTime) and `RotScaleTrans` map. If
   * not, there will only be a `RotScaleTrans` map.
   *
   * \see IncludeDistortedMapType
   */
  template <domain::ObjectLabel Object>
  MapType<Frame::Grid, Frame::Inertial> grid_to_inertial_map(
      const IncludeDistortedMapType& include_distorted_map,
      bool use_rigid_map) const;

  // Names are public because they need to be used when constructing maps in the
  // BCO domain creators themselves
  inline static const std::string expansion_name{"Expansion"};
  inline static const std::string expansion_outer_boundary_name{
      "ExpansionOuterBoundary"};
  inline static const std::string rotation_name{"Rotation"};
  inline static const std::string translation_name{"Translation"};
  inline static const std::array<std::string, 2> size_names{{"SizeA", "SizeB"}};
  inline static const std::array<std::string, 2> shape_names{
      {"ShapeA", "ShapeB"}};

 private:
  static size_t get_index(domain::ObjectLabel object);

  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
  std::optional<ExpansionMapOptions> expansion_map_options_{};
  std::optional<RotationMapOptions> rotation_map_options_{};
  std::optional<TranslationMapOptions> translation_map_options_{};
  std::optional<ShapeMapOptions<domain::ObjectLabel::A>> shape_options_A_{};
  std::optional<ShapeMapOptions<domain::ObjectLabel::B>> shape_options_B_{};
  std::array<std::optional<double>, 2> inner_radii_{};

  // Maps
  std::optional<Expansion> expansion_map_{};
  std::optional<Rotation> rotation_map_{};
  std::optional<std::pair<RotScaleTrans, RotScaleTrans>> rot_scale_trans_map_{};
  using ShapeMapType =
      tmpl::conditional_t<IsCylindrical, std::array<std::optional<Shape>, 2>,
                          std::array<std::array<std::optional<Shape>, 6>, 2>>;
  ShapeMapType shape_maps_{};

  // helper function that creates the functions of time used by the worldtube
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
  create_worldtube_functions_of_time() const;
};
}  // namespace domain::creators::bco
