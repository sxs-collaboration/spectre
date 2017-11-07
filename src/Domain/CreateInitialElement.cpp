// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CreateInitialElement.hpp"

#include <array>

#include "Domain/Block.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/Orientation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

template <size_t VolumeDim, typename TargetFrame>
Element<VolumeDim> create_initial_element(
    const ElementId<VolumeDim>& element_id,
    const Block<VolumeDim, TargetFrame>& block,
    const std::array<size_t, VolumeDim>& initial_refinement_levels) noexcept {
  const auto& neighbors_of_block = block.neighbors();
  const auto& segment_ids = element_id.segment_ids();

  // Declare two helper lambdas for setting the neighbors of an element
  const auto compute_element_neighbor_in_other_block =
      [&neighbors_of_block, &segment_ids](const Direction<VolumeDim>& direction,
                                          const size_t dim) {
        const auto& block_neighbor = neighbors_of_block.at(direction);
        const auto& orientation = block_neighbor.orientation();
        // TODO(): An illustration of what is happening here would be useful
        auto segment_ids_of_neighbor = orientation.mapped(segment_ids);
        auto& segment_to_flip =
            gsl::at(segment_ids_of_neighbor, orientation.mapped(dim));
        segment_to_flip = segment_to_flip.id_if_flipped();
        return std::make_pair(
            direction,
            Neighbors<VolumeDim>(
                {{ElementId<VolumeDim>{block_neighbor.id(),
                                       std::move(segment_ids_of_neighbor)}}},
                orientation));
      };

  const auto compute_element_neighbor_in_same_block =
      [&element_id, &segment_ids](const Direction<VolumeDim>& direction,
                                  const size_t dim, const int sign) {
        auto segment_ids_of_neighbor = segment_ids;
        const auto index = static_cast<int>(gsl::at(segment_ids, dim).index());
        gsl::at(segment_ids_of_neighbor, dim) =
            SegmentId(gsl::at(segment_ids, dim).refinement_level(),
                      static_cast<size_t>(index + sign));
        return std::make_pair(
            direction, Neighbors<VolumeDim>(
                           {{ElementId<VolumeDim>{element_id.block_id(),
                                                  segment_ids_of_neighbor}}},
                           Orientation<VolumeDim>{}));
      };

  typename Element<VolumeDim>::Neighbors_t neighbors_of_element;
  for (size_t d = 0; d < VolumeDim; ++d) {
    const auto index = gsl::at(segment_ids, d).index();
    ASSERT(gsl::at(segment_ids, d).refinement_level() ==
               gsl::at(initial_refinement_levels, d),
           "element id = " << element_id << " and refinement levels = "
                           << initial_refinement_levels);
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
    if (index == two_to_the(gsl::at(initial_refinement_levels, d)) - 1 and
        1 == neighbors_of_block.count(upper_direction)) {
      neighbors_of_element.emplace(
          compute_element_neighbor_in_other_block(upper_direction, d));
    } else if (index != two_to_the(gsl::at(initial_refinement_levels, d)) - 1) {
      neighbors_of_element.emplace(
          compute_element_neighbor_in_same_block(upper_direction, d, 1));
    }
  }
  return Element<VolumeDim>(ElementId<VolumeDim>(element_id),
                            std::move(neighbors_of_element));
}

template Element<1> create_initial_element<1, Frame::Grid>(
    const ElementId<1>&, const Block<1, Frame::Grid>&,
    const std::array<size_t, 1>&) noexcept;
template Element<2> create_initial_element<2, Frame::Grid>(
    const ElementId<2>&, const Block<2, Frame::Grid>&,
    const std::array<size_t, 2>&) noexcept;
template Element<3> create_initial_element<3, Frame::Grid>(
    const ElementId<3>&, const Block<3, Frame::Grid>&,
    const std::array<size_t, 3>&) noexcept;

template Element<1> create_initial_element<1, Frame::Inertial>(
    const ElementId<1>&, const Block<1, Frame::Inertial>&,
    const std::array<size_t, 1>&) noexcept;
template Element<2> create_initial_element<2, Frame::Inertial>(
    const ElementId<2>&, const Block<2, Frame::Inertial>&,
    const std::array<size_t, 2>&) noexcept;
template Element<3> create_initial_element<3, Frame::Inertial>(
    const ElementId<3>&, const Block<3, Frame::Inertial>&,
    const std::array<size_t, 3>&) noexcept;
