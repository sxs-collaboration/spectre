// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Amr/Helpers.hpp"

#include "Domain/Structure/Direction.hpp"       // IWYU pragma: keep
#include "Domain/Structure/ElementId.hpp"       // IWYU pragma: keep
#include "Domain/Structure/OrientationMap.hpp"  // IWYU pragma: keep
#include "Domain/Structure/SegmentId.hpp"       // IWYU pragma: keep
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace amr {
template <size_t VolumeDim>
std::array<size_t, VolumeDim> desired_refinement_levels(
    const ElementId<VolumeDim>& id,
    const std::array<amr::Flag, VolumeDim>& flags) {
  std::array<size_t, VolumeDim> result{};

  for (size_t d = 0; d < VolumeDim; ++d) {
    ASSERT(amr::Flag::Undefined != gsl::at(flags, d),
           "Undefined amr::Flag in dimension " << d);
    gsl::at(result, d) = id.segment_id(d).refinement_level();
    if (amr::Flag::Join == gsl::at(flags, d)) {
      --gsl::at(result, d);
    } else if (amr::Flag::Split == gsl::at(flags, d)) {
      ++gsl::at(result, d);
    }
  }
  return result;
}

template <size_t VolumeDim>
std::array<size_t, VolumeDim> desired_refinement_levels_of_neighbor(
    const ElementId<VolumeDim>& neighbor_id,
    const std::array<amr::Flag, VolumeDim>& neighbor_flags,
    const OrientationMap<VolumeDim>& orientation) {
  if (orientation.is_aligned()) {
    return desired_refinement_levels(neighbor_id, neighbor_flags);
  }
  std::array<size_t, VolumeDim> result{};
  for (size_t d = 0; d < VolumeDim; ++d) {
    ASSERT(amr::Flag::Undefined != gsl::at(neighbor_flags, d),
           "Undefined amr::Flag in dimension " << d);
    const size_t mapped_dim = orientation(d);
    gsl::at(result, d) = neighbor_id.segment_id(mapped_dim).refinement_level();
    if (amr::Flag::Join == gsl::at(neighbor_flags, mapped_dim)) {
      --gsl::at(result, d);
    } else if (amr::Flag::Split == gsl::at(neighbor_flags, mapped_dim)) {
      ++gsl::at(result, d);
    }
  }
  return result;
}

template <size_t VolumeDim>
bool has_potential_sibling(const ElementId<VolumeDim>& element_id,
                           const Direction<VolumeDim>& direction) {
  return direction.side() ==
         element_id.segment_id(direction.dimension()).side_of_sibling();
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template std::array<size_t, DIM(data)> desired_refinement_levels<DIM(data)>( \
      const ElementId<DIM(data)>&, const std::array<amr::Flag, DIM(data)>&);   \
  template std::array<size_t, DIM(data)>                                       \
  desired_refinement_levels_of_neighbor<DIM(data)>(                            \
      const ElementId<DIM(data)>&, const std::array<amr::Flag, DIM(data)>&,    \
      const OrientationMap<DIM(data)>&);                                       \
  template bool has_potential_sibling(const ElementId<DIM(data)>& element_id,  \
                                      const Direction<DIM(data)>& direction);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
}  // namespace amr
