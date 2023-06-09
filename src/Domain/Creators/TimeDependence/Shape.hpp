// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Shape.hpp"
#include "Domain/Creators/TimeDependence/GenerateCoordinateMap.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime
namespace domain::CoordinateMaps::TimeDependent {
class Shape;
}  // namespace domain::CoordinateMaps::TimeDependent
namespace domain {
template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain

namespace Frame {
struct Distorted;
struct Grid;
struct Inertial;
}  // namespace Frame
/// \endcond

namespace domain::creators::time_dependence {
/*!
 * \brief A Shape whose inner surface conforms to a surface of constant
 * Boyer-Lindquist radius, in Kerr-Schild coordinates as given by
 * domain::CoordinateMaps::TimeDependent::Shape.
 *
 * \details This TimeDependence is suitable for use on a spherical shell,
 * where LMax is the number of l and m spherical harmonics to use in
 * approximating the Kerr horizon, of mass `Mass` and spin `Spin`. The value
 * of the Boyer-Lindquist radius to which the inner surface conforms is given
 * by the value of `inner_radius`. If the user wants the inner surface of the
 * Shape to conform to a Kerr horizon for a given mass and spin, `inner_radius`
 * should be the Boyer-Lindquist radius of the outer horizon.
 *
 * The actual shape map that is applied will go from the Grid to the Distorted
 * frame, and then an identity map will go from the Distorted to Inertial frame.
 * The reasoning behind this is because in basically all use cases (BBH), the
 * shape map will go to the Distorted frame only and other maps will go from the
 * Distorted frame to the Inertial frame.
 *
 * \note The quantities stored in the FunctionOfTime are not the
 * complex spherical-harmonic coefficients \f$\lambda_{lm}(t)\f$, but
 * instead are the real-valued SPHEREPACK coefficients
 * \f$a_{lm}(t)\f$ and \f$b_{lm}(t)\f$ used by Spherepack.  The
 * relationship between these two sets of coefficients is
 * \f{align}
 * a_{l0} & = \sqrt{\frac{2}{\pi}}\lambda_{l0}&\qquad l\geq 0,\\
 * a_{lm} & = (-1)^m\sqrt{\frac{2}{\pi}} \mathrm{Re}(\lambda_{lm})
 * &\qquad l\geq 1, m\geq 1, \\
 * b_{lm} & = (-1)^m\sqrt{\frac{2}{\pi}} \mathrm{Im}(\lambda_{lm})
 * &\qquad l\geq 1, m\geq 1.
 * \f}
 * See domain::CoordinateMaps::TimeDependent::Shape for more details.
 *
 * \note To use this time dependence with the `control_system::system::Shape`
 * control system, you must choose the same \tparam Label that the control
 * system is using.
 */
template <domain::ObjectLabel Label>
class Shape final : public TimeDependence<3> {
 private:
  using ShapeMap = domain::CoordinateMaps::TimeDependent::Shape;
  using Identity = domain::CoordinateMaps::Identity<3>;
  using GridToInertialMap =
      detail::generate_coordinate_map_t<Frame::Grid, Frame::Inertial,
                                        tmpl::list<ShapeMap>>;
  using GridToDistortedMap =
      detail::generate_coordinate_map_t<Frame::Grid, Frame::Distorted,
                                        tmpl::list<ShapeMap>>;
  using DistortedToInertialMap =
      detail::generate_coordinate_map_t<Frame::Distorted, Frame::Inertial,
                                        tmpl::list<Identity>>;

 public:
  using maps_list = tmpl::list<
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, ShapeMap>,
      domain::CoordinateMap<Frame::Grid, Frame::Distorted, ShapeMap>,
      domain::CoordinateMap<Frame::Distorted, Frame::Inertial, Identity>>;

  static constexpr size_t mesh_dim = 3;

  static std::string name() { return "Shape"s + get_output(Label); }

  /// \brief The initial time of the function of time.
  struct InitialTime {
    using type = double;
    static constexpr Options::String help = {
        "The initial time of the function of time"};
  };
  /// \brief The max angular resolution `l` of the Shape.
  struct LMax {
    using type = size_t;
    static constexpr Options::String help = {
        "The max l value of the Ylms used by the Shape map."};
  };
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
  /// \brief Center for the Shape map
  struct Center {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {"Center for the Shape map."};
  };
  /// \brief The inner radius of the Shape map, the radius at which
  /// to begin applying the map.
  struct InnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "The inner radius of the Shape map."};
  };
  /// \brief The outer radius of the Shape map, beyond which
  /// it is no longer applied.
  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {
        "The outer radius of the Shape map."};
  };

  using options = tmpl::list<InitialTime, LMax, Mass, Spin, Center, InnerRadius,
                             OuterRadius>;

  static constexpr Options::String help = {
      "Creates a Shape that conforms to a Kerr horizon of given mass and "
      "spin."};

  Shape() = default;
  ~Shape() override = default;
  Shape(const Shape&) = delete;
  Shape(Shape&&) = default;
  Shape& operator=(const Shape&) = delete;
  Shape& operator=(Shape&&) = default;

  Shape(double initial_time, size_t l_max, double mass,
        std::array<double, 3> spin, std::array<double, 3> center,
        double inner_radius, double outer_radius,
        const Options::Context& context = {});

  auto get_clone() const -> std::unique_ptr<TimeDependence<mesh_dim>> override;

  auto block_maps_grid_to_inertial(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Inertial, mesh_dim>>> override;

  auto block_maps_grid_to_distorted(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Distorted, mesh_dim>>> override;

  auto block_maps_distorted_to_inertial(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Distorted, Frame::Inertial, mesh_dim>>> override;

  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  template <domain::ObjectLabel OtherLabel>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const Shape<OtherLabel>& lhs,
                         const Shape<OtherLabel>& rhs);

  using TransitionFunction = domain::CoordinateMaps::
      ShapeMapTransitionFunctions::ShapeMapTransitionFunction;

  GridToInertialMap grid_to_inertial_map() const;
  GridToDistortedMap grid_to_distorted_map() const;
  static DistortedToInertialMap distorted_to_inertial_map();

  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
  size_t l_max_{2};
  double mass_{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, 3> spin_{
      make_array<3>(std::numeric_limits<double>::signaling_NaN())};
  std::array<double, 3> center_{
      make_array<3>(std::numeric_limits<double>::signaling_NaN())};
  inline static const std::string function_of_time_name_{"Shape" +
                                                         get_output(Label)};
  double inner_radius_{std::numeric_limits<double>::signaling_NaN()};
  double outer_radius_{std::numeric_limits<double>::signaling_NaN()};
  std::unique_ptr<TransitionFunction> transition_func_;
};

template <domain::ObjectLabel Label>
bool operator!=(const Shape<Label>& lhs, const Shape<Label>& rhs);
}  // namespace domain::creators::time_dependence
