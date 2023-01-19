// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Amr/Helpers.hpp"

#include <array>
#include <boost/rational.hpp>
#include <cstddef>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"

namespace amr::domain {
template <size_t VolumeDim>
std::array<size_t, VolumeDim> desired_refinement_levels(
    const ElementId<VolumeDim>& id, const std::array<Flag, VolumeDim>& flags) {
  std::array<size_t, VolumeDim> result{};

  for (size_t d = 0; d < VolumeDim; ++d) {
    ASSERT(Flag::Undefined != gsl::at(flags, d),
           "Undefined Flag in dimension " << d);
    gsl::at(result, d) = id.segment_id(d).refinement_level();
    if (Flag::Join == gsl::at(flags, d)) {
      --gsl::at(result, d);
    } else if (Flag::Split == gsl::at(flags, d)) {
      ++gsl::at(result, d);
    }
  }
  return result;
}

template <size_t VolumeDim>
std::array<size_t, VolumeDim> desired_refinement_levels_of_neighbor(
    const ElementId<VolumeDim>& neighbor_id,
    const std::array<Flag, VolumeDim>& neighbor_flags,
    const OrientationMap<VolumeDim>& orientation) {
  if (orientation.is_aligned()) {
    return desired_refinement_levels(neighbor_id, neighbor_flags);
  }
  std::array<size_t, VolumeDim> result{};
  for (size_t d = 0; d < VolumeDim; ++d) {
    ASSERT(Flag::Undefined != gsl::at(neighbor_flags, d),
           "Undefined Flag in dimension " << d);
    const size_t mapped_dim = orientation(d);
    gsl::at(result, d) = neighbor_id.segment_id(mapped_dim).refinement_level();
    if (Flag::Join == gsl::at(neighbor_flags, mapped_dim)) {
      --gsl::at(result, d);
    } else if (Flag::Split == gsl::at(neighbor_flags, mapped_dim)) {
      ++gsl::at(result, d);
    }
  }
  return result;
}

template <size_t VolumeDim>
boost::rational<size_t> fraction_of_block_volume(
    const ElementId<VolumeDim>& element_id) {
  const auto& segment_ids = element_id.segment_ids();
  size_t sum_of_refinement_levels = 0;
  for (const auto& segment_id : segment_ids) {
    sum_of_refinement_levels += segment_id.refinement_level();
  }
  return {1, two_to_the(sum_of_refinement_levels)};
}

template <size_t VolumeDim>
bool has_potential_sibling(const ElementId<VolumeDim>& element_id,
                           const Direction<VolumeDim>& direction) {
  return direction.side() ==
         element_id.segment_id(direction.dimension()).side_of_sibling();
}

template <size_t VolumeDim>
ElementId<VolumeDim> id_of_parent(const ElementId<VolumeDim>& element_id,
                                  const std::array<Flag, VolumeDim>& flags) {
  using ::operator<<;
  ASSERT(alg::count(flags, Flag::Join) > 0,
         "Element " << element_id << " is not joining given flags " << flags);
  ASSERT(alg::count(flags, Flag::Split) == 0,
         "Splitting and joining an Element is not supported");
  auto segment_ids = element_id.segment_ids();
  for (size_t d = 0; d < VolumeDim; ++d) {
    if (gsl::at(flags, d) == Flag::Join) {
      gsl::at(segment_ids, d) = element_id.segment_id(d).id_of_parent();
    }
  }
  return {element_id.block_id(), std::move(segment_ids),
          element_id.grid_index()};
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template std::array<size_t, DIM(data)> desired_refinement_levels<DIM(data)>( \
      const ElementId<DIM(data)>&, const std::array<Flag, DIM(data)>&);        \
  template std::array<size_t, DIM(data)>                                       \
  desired_refinement_levels_of_neighbor<DIM(data)>(                            \
      const ElementId<DIM(data)>&, const std::array<Flag, DIM(data)>&,         \
      const OrientationMap<DIM(data)>&);                                       \
  template boost::rational<size_t> fraction_of_block_volume<DIM(data)>(        \
      const ElementId<DIM(data)>& element_id);                                 \
  template bool has_potential_sibling(const ElementId<DIM(data)>& element_id,  \
                                      const Direction<DIM(data)>& direction);  \
  template ElementId<DIM(data)> id_of_parent(                                  \
      const ElementId<DIM(data)>& element_id,                                  \
      const std::array<amr::domain::Flag, DIM(data)>& flags);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
}  // namespace amr::domain
