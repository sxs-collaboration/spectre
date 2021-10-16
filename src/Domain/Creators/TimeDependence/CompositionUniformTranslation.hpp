// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/TimeDependence/GenerateCoordinateMap.hpp"
#include "Domain/Creators/TimeDependence/OptionTags.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Options/Options.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace Frame {
struct Grid;
struct Inertial;
}  // namespace Frame

namespace domain {
namespace creators {
namespace time_dependence {
/*!
 * \brief A TimeDependence that is a composition of two `UniformTranslation`s.
 */
template <size_t MeshDim>
class CompositionUniformTranslation final : public TimeDependence<MeshDim> {
 private:
  using TranslationMap =
      domain::CoordinateMaps::TimeDependent::Translation<MeshDim>;

 public:
  using CoordMap = domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                         TranslationMap, TranslationMap>;
  // using CoordMap = detail::generate_coordinate_map_t<tmpl::flatten<
  //    tmpl::list<typename UniformTranslation<MeshDim>::maps_list,
  //               typename UniformTranslation<MeshDim>::maps_list>>>;
  // static_assert(std::is_same_v<CoordMap, int> , "Hello I failed.");
  using maps_list = tmpl::list<CoordMap>;
  static constexpr Options::String help = {
      "A composition of two UniformTranslations."};

  using options = tmpl::list<
      OptionTags::TimeDependenceCompositionTag<UniformTranslation<MeshDim>>,
      OptionTags::TimeDependenceCompositionTag<UniformTranslation<MeshDim>, 1>>;

  CompositionUniformTranslation() = default;
  ~CompositionUniformTranslation() override = default;
  CompositionUniformTranslation(const CompositionUniformTranslation&) = default;
  CompositionUniformTranslation& operator=(
      const CompositionUniformTranslation&) = default;
  CompositionUniformTranslation(CompositionUniformTranslation&&) = default;
  CompositionUniformTranslation& operator=(CompositionUniformTranslation&&) =
      default;

  explicit CompositionUniformTranslation(
      std::unique_ptr<TimeDependence<MeshDim>> uniform_translation0,
      std::unique_ptr<TimeDependence<MeshDim>> uniform_translation1);

  /// Constructor for copying the composition time dependence. Internally
  /// performs all the copying necessary to deal with the functions of time.
  CompositionUniformTranslation(
      CoordMap coord_map,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time);

  auto get_clone() const -> std::unique_ptr<TimeDependence<MeshDim>> override;

  auto block_maps(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Inertial, MeshDim>>> override;

  auto functions_of_time() const -> std::unordered_map<
      std::string,
      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  CoordMap coord_map_;

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time_;
};
}  // namespace time_dependence
}  // namespace creators
}  // namespace domain
