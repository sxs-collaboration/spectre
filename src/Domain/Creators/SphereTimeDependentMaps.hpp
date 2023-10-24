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
#include "Domain/CoordinateMaps/TimeDependent/Shape.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
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
 * \brief Mass and spin necessary for calculating the \f$ Y_{lm} \f$
 * coefficients of a Kerr horizon of certain Boyer-Lindquist radius for the
 * shape map of the Sphere domain creator.
 */
struct KerrSchildFromBoyerLindquist {
  /// \brief The mass of the Kerr black hole.
  struct Mass {
    using type = double;
    static constexpr Options::String help = {"The mass of the Kerr BH."};
  };
  /// \brief The dimensionless spin of the Kerr black hole.
  struct Spin {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "The dim'less spin of the Kerr BH."};
  };

  using options = tmpl::list<Mass, Spin>;

  static constexpr Options::String help = {
      "Conform to an ellipsoid of constant Boyer-Lindquist radius in "
      "Kerr-Schild coordinates. This Boyer-Lindquist radius is chosen as the "
      "value of the 'InnerRadius'. To conform to the outer Kerr horizon, "
      "choose an 'InnerRadius' of r_+ = M + sqrt(M^2-a^2)."};

  double mass{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, 3> spin{std::numeric_limits<double>::signaling_NaN(),
                             std::numeric_limits<double>::signaling_NaN(),
                             std::numeric_limits<double>::signaling_NaN()};
};

// Label for shape map options
struct Spherical {};

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
  using TranslationMap = domain::CoordinateMaps::TimeDependent::Translation<3>;

  template <typename SourceFrame, typename TargetFrame>
  using IdentityForComposition =
      domain::CoordinateMap<SourceFrame, TargetFrame, IdentityMap>;
  using GridToDistortedComposition =
      domain::CoordinateMap<Frame::Grid, Frame::Distorted, ShapeMap>;
  using GridToInertialComposition =
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, ShapeMap,
                            TranslationMap>;
  using GridToInertialSimple =
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, TranslationMap>;
  using DistortedToInertialComposition =
      domain::CoordinateMap<Frame::Distorted, Frame::Inertial, TranslationMap>;

 public:
  using maps_list =
      tmpl::list<IdentityForComposition<Frame::Grid, Frame::Inertial>,
                 IdentityForComposition<Frame::Grid, Frame::Distorted>,
                 IdentityForComposition<Frame::Distorted, Frame::Inertial>,
                 GridToDistortedComposition, GridToInertialSimple,
                 GridToInertialComposition, DistortedToInertialComposition>;

  /// \brief The initial time of the functions of time.
  struct InitialTime {
    using type = double;
    static constexpr Options::String help = {
        "The initial time of the functions of time"};
  };

  struct TranslationMapOptions {
    static std::string name() { return "TranslationMap"; }
    static constexpr Options::String help = {
        "Options for a time-dependent translation map in the inner-most shell "
        "of the domain."};
  };

  struct ShapeMapOptions {
    using type = ShapeMapOptions;
    static std::string name() { return "ShapeMap"; }
    static constexpr Options::String help = {
        "Options for a time-dependent shape map in the inner-most shell of the "
        "domain."};

    struct LMax {
      using type = size_t;
      static constexpr Options::String help = {
          "Initial LMax for the shape map."};
    };

    struct InitialValues {
      using type =
          Options::Auto<std::variant<KerrSchildFromBoyerLindquist>, Spherical>;
      static constexpr Options::String help = {
          "Initial Ylm coefficients for the shape map. Specify 'Spherical' for "
          "all coefficients to be initialized to zero."};
    };

    using options = tmpl::list<LMax, InitialValues>;

    size_t l_max{};
    std::optional<std::variant<KerrSchildFromBoyerLindquist>> initial_values{};
  };

  struct InitialVelocity {
    static std::string name() { return "InitialVelocity"; }
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Velocity of the translation map."};
    using group = TranslationMapOptions;
  };

  using options = tmpl::list<InitialTime, ShapeMapOptions, InitialVelocity>;
  static constexpr Options::String help{
      "The options for all the hard-coded time dependent maps in the Sphere "
      "domain."};

  TimeDependentMapOptions() = default;

  TimeDependentMapOptions(
      double initial_time, const ShapeMapOptions& shape_map_options,
      const std::array<double, 3>& initial_translation_velocity);

  /*!
   * \brief Create the function of time map using the options that were
   * provided to this class.
   *
   * Currently, this will add:
   *
   * - Size: `PiecewisePolynomial<3>`
   * - Shape: `PiecewisePolynomial<2>`
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
   * - Translation: `Translation`
   */
  void build_maps(
      const std::array<double, 3>& center, double inner_radius,
      double outer_radius,
      std::optional<std::pair<double, double>> translation_transition_radii);

  /*!
   * \brief This will construct the map from `Frame::Distorted` to
   * `Frame::Inertial`.
   *
   * If the argument `include_distorted_map` is true, then this will be a
   * translation map. If it is false, then this returns `nullptr`.
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
   * \brief This will construct the map from `Frame::Grid` to `Frame::Inertial`.
   *
   * If the argument `include_distorted_map` is true, then this map will have a
   * `Shape` map (with a size function of time). If it is false, then there will
   * only be an identity map.
   */
  MapType<Frame::Grid, Frame::Inertial> grid_to_inertial_map(
      bool include_distorted_map, bool use_rigid_translation) const;

  inline static const std::string size_name{"Size"};
  inline static const std::string shape_name{"Shape"};
  inline static const std::string translation_name{"Translation"};

 private:
  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
  size_t initial_l_max_{};
  ShapeMap shape_map_{};
  TranslationMap rigid_translation_map_{};
  TranslationMap translation_map_{};
  std::optional<std::variant<KerrSchildFromBoyerLindquist>>
      initial_shape_values_{};
  std::array<double, 3> initial_translation_velocity_{
      std::numeric_limits<std::array<double, 3>>::signaling_NaN()};
};
}  // namespace domain::creators::sphere
