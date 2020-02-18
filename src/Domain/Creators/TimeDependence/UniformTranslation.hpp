// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Translation.hpp"
#include "Domain/Creators/TimeDependence/GenerateCoordinateMap.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
namespace CoordMapsTimeDependent {
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
template <typename Map1, typename Map2>
class ProductOf2Maps;
}  // namespace CoordMapsTimeDependent
}  // namespace domain

namespace Frame {
struct Grid;
struct Inertial;
}  // namespace Frame
/// \endcond

namespace domain {
namespace creators {
namespace time_dependence {
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
 */
template <size_t MeshDim>
class UniformTranslation final : public TimeDependence<MeshDim> {
 private:
  using Translation = domain::CoordMapsTimeDependent::Translation;

 public:
  static constexpr size_t mesh_dim = MeshDim;

  /// \brief The initial time of the functions of time.
  struct InitialTime {
    using type = double;
    static constexpr OptionString help = {
        "The initial time of the functions of time"};
  };
  /// \brief The \f$x\f$-, \f$y\f$-, and \f$z\f$-velocity.
  struct Velocity {
    using type = std::array<double, MeshDim>;
    static constexpr OptionString help = {"The velocity of the map."};
  };
  /// \brief The names of the functions of times to be added to the added to the
  /// DataBox.
  ///
  /// The defaults are `"TranslationX", "TranslationY", "TranslationZ"`.
  struct FunctionOfTimeNames {
    using type = std::array<std::string, MeshDim>;
    static constexpr OptionString help = {"Names of the functions of time."};
    static type default_value() noexcept {
      return UniformTranslation::default_function_names();
    }
  };

  using MapForComposition =
      detail::generate_coordinate_map_t<tmpl::list<tmpl::conditional_t<
          MeshDim == 1, Translation,
          tmpl::conditional_t<MeshDim == 2,
                              domain::CoordMapsTimeDependent::ProductOf2Maps<
                                  Translation, Translation>,
                              domain::CoordMapsTimeDependent::ProductOf3Maps<
                                  Translation, Translation, Translation>>>>>;

  using options = tmpl::list<InitialTime, Velocity, FunctionOfTimeNames>;

  static constexpr OptionString help = {
      "A spatially uniform translation initialized with a constant velocity."};

  UniformTranslation() = default;
  ~UniformTranslation() override = default;
  UniformTranslation(const UniformTranslation&) = delete;
  UniformTranslation(UniformTranslation&&) noexcept = default;
  UniformTranslation& operator=(const UniformTranslation&) = delete;
  UniformTranslation& operator=(UniformTranslation&&) noexcept = default;

  UniformTranslation(double initial_time,
                     const std::array<double, MeshDim>& velocity,
                     std::array<std::string, MeshDim> functions_of_time_names =
                         default_function_names()) noexcept;

  auto get_clone() const noexcept
      -> std::unique_ptr<TimeDependence<MeshDim>> override;

  auto block_maps(size_t number_of_blocks) const noexcept
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Inertial, MeshDim>>> override;

  auto functions_of_time() const noexcept -> std::unordered_map<
      std::string,
      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

  /// Returns the map for each block to be used in a composition of
  /// `TimeDependence`s.
  MapForComposition map_for_composition() const noexcept;

 private:
  static std::array<std::string, MeshDim> default_function_names() noexcept;

  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const UniformTranslation<LocalDim>& lhs,
                         const UniformTranslation<LocalDim>& rhs) noexcept;

  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, MeshDim> velocity_{};
  std::array<std::string, MeshDim> functions_of_time_names_{};
};

template <size_t Dim>
bool operator==(const UniformTranslation<Dim>& lhs,
                const UniformTranslation<Dim>& rhs) noexcept;

template <size_t Dim>
bool operator!=(const UniformTranslation<Dim>& lhs,
                const UniformTranslation<Dim>& rhs) noexcept;
}  // namespace time_dependence
}  // namespace creators
}  // namespace domain
