// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::creators::time_dependent_options {
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

/// Label for shape map options
struct Spherical {};

/*!
 * \brief Class to be used as an option for initializing shape map coefficients.
 *
 * \tparam IncludeTransitionEndsAtCube This is mainly added for the
 * `domain::creators::BinaryCompactObject` domain.
 * \tparam Object Which object that this shape map represents. Use
 * `domain::ObjectLabel::None` if there is only a single object in your
 * simulation.
 */
template <bool IncludeTransitionEndsAtCube, domain::ObjectLabel Object>
struct ShapeMapOptions {
  using type = Options::Auto<ShapeMapOptions, Options::AutoLabel::None>;
  static std::string name() { return "ShapeMap" + get_output(Object); }
  static constexpr Options::String help = {
      "Options for a time-dependent distortion (shape) map about the "
      "specified object. Specify 'None' to not use this map."};

  struct LMax {
    using type = size_t;
    static constexpr Options::String help = {
        "LMax used for the number of spherical harmonic coefficients of the "
        "distortion map."};
  };

  struct InitialValues {
    using type =
        Options::Auto<std::variant<KerrSchildFromBoyerLindquist>, Spherical>;
    static constexpr Options::String help = {
        "Initial Ylm coefficients for the shape map. Specify 'Spherical' for "
        "all coefficients to be initialized to zero."};
  };

  struct SizeInitialValues {
    using type = Options::Auto<std::array<double, 3>>;
    static constexpr Options::String help = {
        "Initial value and two derivatives of the 00 coefficient. Specify "
        "'Auto' to use the 00 coefficient specified in the 'InitialValues' "
        "option."};
  };

  struct TransitionEndsAtCube {
    using type = bool;
    static constexpr Options::String help = {
        "If 'true', the shape map transition function will be 0 at the cubical "
        "boundary around the object. If 'false' the transition function will "
        "be 0 at the outer radius of the inner sphere around the object"};
  };

  using common_options = tmpl::list<LMax, InitialValues, SizeInitialValues>;

  using options =
      tmpl::conditional_t<IncludeTransitionEndsAtCube,
                          tmpl::push_back<common_options, TransitionEndsAtCube>,
                          common_options>;

  size_t l_max{};
  std::optional<std::variant<KerrSchildFromBoyerLindquist>> initial_values{};
  std::optional<std::array<double, 3>> initial_size_values{};
  bool transition_ends_at_cube{false};
};

template <bool IncludeTransitionEndsAtCube, domain::ObjectLabel Object>
std::pair<std::array<DataVector, 3>, std::array<DataVector, 4>>
initial_shape_and_size_funcs(
    const ShapeMapOptions<IncludeTransitionEndsAtCube, Object>& shape_options,
    double inner_radius);
}  // namespace domain::creators::time_dependent_options
