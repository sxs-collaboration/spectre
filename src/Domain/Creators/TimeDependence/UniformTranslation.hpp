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
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/TimeDependence/GenerateCoordinateMap.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Options/String.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime
namespace domain::CoordinateMaps::TimeDependent {
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
template <typename Map1, typename Map2>
class ProductOf2Maps;
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
 * \brief A uniform translation in the \f$x-, y-\f$ and \f$z-\f$direction.
 *
 * The coordinates are adjusted according to:
 *
 * \f{align}{
 * x^i \to x^i + f^i(t)
 * \f}
 *
 * where \f$f^i(t)\f$ are the functions of time.
 *
 * \p Index is used to distinguish multiple `UniformTranslation`s from each
 * other in CompositionUniformTranslation.
 *
 * See the documentation for the constructors below: one constructor
 * takes two velocities, which correspond to two translations: one from
 * Frame::Grid to Frame::Distorted, and the other from Frame::Distorted to
 * Frame::Inertial.
 */
template <size_t MeshDim, size_t Index>
class UniformTranslation final : public TimeDependence<MeshDim> {
 private:
  using TranslationMap =
      domain::CoordinateMaps::TimeDependent::Translation<MeshDim>;

 public:
  using maps_list = tmpl::list<
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, TranslationMap>,
      domain::CoordinateMap<Frame::Grid, Frame::Distorted, TranslationMap>,
      domain::CoordinateMap<Frame::Distorted, Frame::Inertial, TranslationMap>,
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, TranslationMap,
                            TranslationMap>>;

  static constexpr size_t mesh_dim = MeshDim;

  /// \brief The initial time of the functions of time.
  struct InitialTime {
    using type = double;
    static constexpr Options::String help = {
        "The initial time of the functions of time"};
  };
  /// \brief The \f$x\f$-, \f$y\f$-, and \f$z\f$-velocity.
  struct Velocity {
    using type = std::array<double, MeshDim>;
    static constexpr Options::String help = {"The velocity of the map."};
  };

  using GridToInertialMapSimple =
      detail::generate_coordinate_map_t<Frame::Grid, Frame::Inertial,
                                        tmpl::list<TranslationMap>>;

  using GridToInertialMapCombined = detail::generate_coordinate_map_t<
      Frame::Grid, Frame::Inertial, tmpl::list<TranslationMap, TranslationMap>>;

  using GridToDistortedMap =
      detail::generate_coordinate_map_t<Frame::Grid, Frame::Distorted,
                                        tmpl::list<TranslationMap>>;

  using DistortedToInertialMap =
      detail::generate_coordinate_map_t<Frame::Distorted, Frame::Inertial,
                                        tmpl::list<TranslationMap>>;

  using options = tmpl::list<InitialTime, Velocity>;

  static constexpr Options::String help = {
      "A spatially uniform translation initialized with a constant velocity."};

  UniformTranslation() = default;
  ~UniformTranslation() override = default;
  UniformTranslation(const UniformTranslation&) = delete;
  UniformTranslation(UniformTranslation&&) = default;
  UniformTranslation& operator=(const UniformTranslation&) = delete;
  UniformTranslation& operator=(UniformTranslation&&) = default;

  /// If UniformTranslation is created using the constructor that
  /// takes a single velocity, then there is no distorted frame (so
  /// block_maps_grid_to_distorted() and
  /// block_maps_distorted_to_inertial() contain nullptrs), and the
  /// given velocity is the one that goes from Frame::Grid to
  /// Frame::Inertial.
  UniformTranslation(double initial_time,
                     const std::array<double, MeshDim>& velocity);

  /// If UniformTranslation is created using the constructor that
  /// takes two velocities, then the first velocity is the one
  /// describing a uniform translation that goes from Frame::Grid to
  /// Frame::Distorted, and the second velocity is the one that
  /// describes a uniform translation that goes from Frame::Distorted
  /// to Frame::Inertial.  In this case there are also two
  /// FunctionsOfTime, one for each of the two translation maps.
  UniformTranslation(
      double initial_time,
      const std::array<double, MeshDim>& velocity_grid_to_distorted,
      const std::array<double, MeshDim>& velocity_distorted_to_inertial);

  auto get_clone() const -> std::unique_ptr<TimeDependence<MeshDim>> override;

  auto block_maps_grid_to_inertial(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Inertial, MeshDim>>> override;

  auto block_maps_grid_to_distorted(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Distorted, MeshDim>>> override;

  auto block_maps_distorted_to_inertial(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Distorted, Frame::Inertial, MeshDim>>> override;

  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

  static std::string name() {
    return "UniformTranslation" + (Index == 0 ? "" : get_output(Index));
  }

 private:
  template <size_t LocalDim, size_t LocalIndex>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const UniformTranslation<LocalDim, LocalIndex>& lhs,
                         const UniformTranslation<LocalDim, LocalIndex>& rhs);

  GridToInertialMapSimple grid_to_inertial_map_simple() const;
  GridToInertialMapCombined grid_to_inertial_map_combined() const;
  GridToDistortedMap grid_to_distorted_map() const;
  DistortedToInertialMap distorted_to_inertial_map() const;

  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
  // If distorted and inertial frames are equal, then
  // velocity_grid_to_distorted_ is the grid to inertial velocity, and
  // velocity_distorted_to_inertial_ is unused.
  std::array<double, MeshDim> velocity_grid_to_distorted_{};
  std::array<double, MeshDim> velocity_distorted_to_inertial_{};
  bool distorted_and_inertial_frames_are_equal_{true};
  inline static const std::string function_of_time_name_grid_to_distorted_{
      "Translation" + (Index == 0 ? "" : get_output(Index))};
  inline static const std::string function_of_time_name_distorted_to_inertial_{
      "TranslationDistortedToInertial" + (Index == 0 ? "" : get_output(Index))};
};

template <size_t Dim, size_t Index>
bool operator==(const UniformTranslation<Dim, Index>& lhs,
                const UniformTranslation<Dim, Index>& rhs);

template <size_t Dim, size_t Index>
bool operator!=(const UniformTranslation<Dim, Index>& lhs,
                const UniformTranslation<Dim, Index>& rhs);
}  // namespace domain::creators::time_dependence
