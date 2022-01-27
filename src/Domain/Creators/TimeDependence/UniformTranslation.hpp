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
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
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
 */
template <size_t MeshDim, size_t Index>
class UniformTranslation final : public TimeDependence<MeshDim> {
 private:
  using TranslationMap =
      domain::CoordinateMaps::TimeDependent::Translation<MeshDim>;

 public:
  using maps_list = tmpl::list<
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, TranslationMap>>;

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

  using GridToInertialMap =
      detail::generate_coordinate_map_t<Frame::Grid, Frame::Inertial,
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

  UniformTranslation(double initial_time,
                     const std::array<double, MeshDim>& velocity);

  auto get_clone() const -> std::unique_ptr<TimeDependence<MeshDim>> override;

  auto block_maps_grid_to_inertial(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Inertial, MeshDim>>> override;

  auto block_maps_grid_to_distorted(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Distorted, MeshDim>>> override {
    using ptr_type =
        domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, MeshDim>;
    return std::vector<std::unique_ptr<ptr_type>>(number_of_blocks);
  }

  auto block_maps_distorted_to_inertial(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Distorted, Frame::Inertial, MeshDim>>> override {
    using ptr_type =
        domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, MeshDim>;
    return std::vector<std::unique_ptr<ptr_type>>(number_of_blocks);
  }

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

  GridToInertialMap grid_to_inertial_map() const;

  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, MeshDim> velocity_{};
  inline static const std::string function_of_time_name_{
      "Translation" + (Index == 0 ? "" : get_output(Index))};
};

template <size_t Dim, size_t Index>
bool operator==(const UniformTranslation<Dim, Index>& lhs,
                const UniformTranslation<Dim, Index>& rhs);

template <size_t Dim, size_t Index>
bool operator!=(const UniformTranslation<Dim, Index>& lhs,
                const UniformTranslation<Dim, Index>& rhs);
}  // namespace domain::creators::time_dependence
