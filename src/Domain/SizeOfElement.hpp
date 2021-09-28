// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
template <typename SourceFrame, typename TargetFrame, size_t Dim>
class CoordinateMapBase;
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
}  // namespace domain
template <size_t VolumeDim, typename Frame>
class ElementMap;
namespace domain {
namespace Tags {
template <size_t Dim, typename Frame>
struct ElementMap;
}  // namespace Tags
}  // namespace domain
/// \endcond

/// @{
/*!
 * \ingroup ComputationalDomainGroup
 * \brief Compute the inertial-coordinate size of an element along each of its
 * logical directions.
 *
 * For each logical direction, compute the distance (in inertial coordinates)
 * between the element's lower and upper faces in that logical direction.
 * The distance is measured between centers of the faces, with the centers
 * defined in the logical coordinates.
 * Note that for curved elements, this is an approximate measurement of size.
 *
 * \details
 * Because this quantity is defined in terms of specific coordinates, it is
 * not well represented by a `Tensor`, so we use a `std::array`.
 */
template <size_t VolumeDim>
std::array<double, VolumeDim> size_of_element(
    const ElementMap<VolumeDim, Frame::Inertial>& logical_to_inertial_map);

template <size_t VolumeDim>
std::array<double, VolumeDim> size_of_element(
    const ElementMap<VolumeDim, Frame::Grid>& logical_to_grid_map,
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>&
        grid_to_inertial_map,
    double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time);
/// @}

namespace domain {
namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The inertial-coordinate size of an element along each of its logical
/// directions.
template <size_t VolumeDim>
struct SizeOfElement : db::SimpleTag {
  using type = std::array<double, VolumeDim>;
};

template <size_t VolumeDim>
struct SizeOfElementCompute : db::ComputeTag, SizeOfElement<VolumeDim> {
  using base = SizeOfElement<VolumeDim>;
  using argument_tags =
      tmpl::list<Tags::ElementMap<VolumeDim, Frame::Grid>,
                 CoordinateMaps::Tags::CoordinateMap<VolumeDim, Frame::Grid,
                                                     Frame::Inertial>,
                 ::Tags::Time, domain::Tags::FunctionsOfTime>;
  using return_type = typename base::type;

  static constexpr void function(
      gsl::not_null<std::array<double, VolumeDim>*> result,
      const ::ElementMap<VolumeDim, Frame::Grid>& logical_to_grid_map,
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>&
          grid_to_inertial_map,
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) {
    *result = size_of_element(logical_to_grid_map, grid_to_inertial_map, time,
                              functions_of_time);
  }
};
}  // namespace Tags
}  // namespace domain
