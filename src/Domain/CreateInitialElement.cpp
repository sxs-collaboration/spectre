// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CreateInitialElement.hpp"

#include <utility>

#include "Domain/Block.hpp"          // IWYU pragma: keep
#include "Domain/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Side.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Frame {
struct Grid;      // IWYU pragma: keep
struct Inertial;  // IWYU pragma: keep
}  // namespace Frame

template <size_t VolumeDim, typename TargetFrame>
Element<VolumeDim> create_initial_element(
    const ElementId<VolumeDim>& element_id,
    const Block<VolumeDim, TargetFrame>& block) noexcept {
  const auto& neighbors_of_block = block.neighbors();
  const auto& segment_ids = element_id.segment_ids();

  // Declare two helper lambdas for setting the neighbors of an element
  const auto compute_element_neighbor_in_other_block = [&neighbors_of_block,
                                                        &segment_ids](
      const Direction<VolumeDim>& direction, const size_t dim) {
    const auto& block_neighbor = neighbors_of_block.at(direction);
    const auto& orientation = block_neighbor.orientation();
    // TODO(): An illustration of what is happening here would be useful
    auto segment_ids_of_neighbor = orientation(segment_ids);
    auto& segment_to_flip = gsl::at(segment_ids_of_neighbor, orientation(dim));
    segment_to_flip = segment_to_flip.id_if_flipped();
    return std::make_pair(
        direction,
        Neighbors<VolumeDim>(
            {{ElementId<VolumeDim>{block_neighbor.id(),
                                   std::move(segment_ids_of_neighbor)}}},
            orientation));
  };

  const auto compute_element_neighbor_in_same_block = [&element_id,
                                                       &segment_ids](
      const Direction<VolumeDim>& direction, const size_t dim, const int sign) {
    auto segment_ids_of_neighbor = segment_ids;
    const auto index = static_cast<int>(gsl::at(segment_ids, dim).index());
    gsl::at(segment_ids_of_neighbor, dim) =
        SegmentId(gsl::at(segment_ids, dim).refinement_level(),
                  static_cast<size_t>(index + sign));
    return std::make_pair(
        direction,
        Neighbors<VolumeDim>({{ElementId<VolumeDim>{element_id.block_id(),
                                                    segment_ids_of_neighbor}}},
                             OrientationMap<VolumeDim>{}));
  };

  typename Element<VolumeDim>::Neighbors_t neighbors_of_element;
  for (size_t d = 0; d < VolumeDim; ++d) {
    const auto index = gsl::at(segment_ids, d).index();
    // lower neighbor
    const auto lower_direction = Direction<VolumeDim>{d, Side::Lower};
    if (0 == index and 1 == neighbors_of_block.count(lower_direction)) {
      neighbors_of_element.emplace(
          compute_element_neighbor_in_other_block(lower_direction, d));
    } else if (0 != index) {
      neighbors_of_element.emplace(
          compute_element_neighbor_in_same_block(lower_direction, d, -1));
    }
    // upper neighbor
    const auto upper_direction = Direction<VolumeDim>{d, Side::Upper};
    if (index == two_to_the(gsl::at(segment_ids, d).refinement_level()) - 1 and
        1 == neighbors_of_block.count(upper_direction)) {
      neighbors_of_element.emplace(
          compute_element_neighbor_in_other_block(upper_direction, d));
    } else if (index !=
               two_to_the(gsl::at(segment_ids, d).refinement_level()) - 1) {
      neighbors_of_element.emplace(
          compute_element_neighbor_in_same_block(upper_direction, d, 1));
    }
  }
  return Element<VolumeDim>(ElementId<VolumeDim>(element_id),
                            std::move(neighbors_of_element));
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                  \
  template Element<DIM(data)> create_initial_element<DIM(data), FRAME(data)>( \
      const ElementId<DIM(data)>&,                                            \
      const Block<DIM(data), FRAME(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef DIM
#undef FRAME
#undef INSTANTIATE
/// \endcond
