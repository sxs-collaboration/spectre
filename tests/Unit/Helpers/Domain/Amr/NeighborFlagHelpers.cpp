// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/Domain/Amr/NeighborFlagHelpers.hpp"

#include <array>
#include <vector>

#include "Domain/Amr/Helpers.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim>
std::vector<std::array<::amr::Flag, Dim>> amr_flags();

template <>
std::vector<std::array<::amr::Flag, 1>> amr_flags<1>() {
  // use IncreaseResolution instead of DoNothing to catch any instances of
  // hard-coding DoNothing when the function should be valid for any of
  // DoNothing, DecreaseResolution, or IncreaseResolution
  return std::vector{std::array{::amr::Flag::Join},
                     std::array{::amr::Flag::IncreaseResolution},
                     std::array{::amr::Flag::Split}};
}

template <>
std::vector<std::array<::amr::Flag, 2>> amr_flags<2>() {
  // use IncreaseResolution instead of DoNothing to catch any instances of
  // hard-coding DoNothing when the function should be valid for any of
  // DoNothing, DecreaseResolution, or IncreaseResolution
  return std::vector{
      std::array{::amr::Flag::Join, ::amr::Flag::Join},
      std::array{::amr::Flag::Join, ::amr::Flag::IncreaseResolution},
      std::array{::amr::Flag::IncreaseResolution, ::amr::Flag::Join},
      std::array{::amr::Flag::IncreaseResolution,
                 ::amr::Flag::IncreaseResolution},
      std::array{::amr::Flag::IncreaseResolution, ::amr::Flag::Split},
      std::array{::amr::Flag::Split, ::amr::Flag::IncreaseResolution},
      std::array{::amr::Flag::Split, ::amr::Flag::Split}};
}

template <>
std::vector<std::array<::amr::Flag, 3>> amr_flags<3>() {
  // use IncreaseResolution instead of DoNothing to catch any instances of
  // hard-coding DoNothing when the function should be valid for any of
  // DoNothing, DecreaseResolution, or IncreaseResolution
  return std::vector{
      std::array{::amr::Flag::Join, ::amr::Flag::Join, ::amr::Flag::Join},
      std::array{::amr::Flag::Join, ::amr::Flag::Join,
                 ::amr::Flag::IncreaseResolution},
      std::array{::amr::Flag::Join, ::amr::Flag::IncreaseResolution,
                 ::amr::Flag::Join},
      std::array{::amr::Flag::Join, ::amr::Flag::IncreaseResolution,
                 ::amr::Flag::IncreaseResolution},
      std::array{::amr::Flag::IncreaseResolution, ::amr::Flag::Join,
                 ::amr::Flag::Join},
      std::array{::amr::Flag::IncreaseResolution, ::amr::Flag::Join,
                 ::amr::Flag::IncreaseResolution},
      std::array{::amr::Flag::IncreaseResolution,
                 ::amr::Flag::IncreaseResolution, ::amr::Flag::Join},
      std::array{::amr::Flag::IncreaseResolution,
                 ::amr::Flag::IncreaseResolution,
                 ::amr::Flag::IncreaseResolution},
      std::array{::amr::Flag::IncreaseResolution,
                 ::amr::Flag::IncreaseResolution, ::amr::Flag::Split},
      std::array{::amr::Flag::IncreaseResolution, ::amr::Flag::Split,
                 ::amr::Flag::IncreaseResolution},
      std::array{::amr::Flag::IncreaseResolution, ::amr::Flag::Split,
                 ::amr::Flag::Split},
      std::array{::amr::Flag::Split, ::amr::Flag::IncreaseResolution,
                 ::amr::Flag::IncreaseResolution},
      std::array{::amr::Flag::Split, ::amr::Flag::IncreaseResolution,
                 ::amr::Flag::Split},
      std::array{::amr::Flag::Split, ::amr::Flag::Split,
                 ::amr::Flag::IncreaseResolution},
      std::array{::amr::Flag::Split, ::amr::Flag::Split, ::amr::Flag::Split}};
}

template <size_t Dim>
bool are_valid_neighbor_flags(
    const ElementId<Dim>& element_id,
    const std::array<::amr::Flag, Dim>& element_flags,
    const ElementId<Dim>& neighbor_id,
    const std::array<::amr::Flag, Dim>& neighbor_flags,
    const OrientationMap<Dim>& orientation_of_neighbor = {}) {
  if (element_id == neighbor_id) {
    return element_flags == neighbor_flags;
  }
  for (size_t d = 0; d < Dim; ++d) {
    if (neighbor_id.segment_id(d) == SegmentId{0, 0} and
        gsl::at(neighbor_flags, d) == amr::Flag::Join) {
      return false;
    }
  }
  const auto element_desired_levels =
      desired_refinement_levels(element_id, element_flags);
  const auto neighbor_desired_levels = desired_refinement_levels_of_neighbor(
      neighbor_id, neighbor_flags, orientation_of_neighbor);
  for (size_t d = 0; d < Dim; ++d) {
    if ((gsl::at(element_desired_levels, d) >
         gsl::at(neighbor_desired_levels, d) + 1) or
        (gsl::at(neighbor_desired_levels, d) >
         gsl::at(element_desired_levels, d) + 1)) {
      return false;
    }
    if (element_id.block_id() == neighbor_id.block_id()) {
      const auto& element_segment = element_id.segment_id(d);
      const auto& neighbor_segment = neighbor_id.segment_id(d);
      if (element_segment.refinement_level() == 0 or
          neighbor_segment.refinement_level() == 0) {
        continue;
      }
      if (element_segment.endpoint(element_segment.side_of_sibling()) ==
          neighbor_segment.endpoint(neighbor_segment.side_of_sibling())) {
        if (gsl::at(element_flags, d) == ::amr::Flag::Join) {
          if (element_desired_levels != neighbor_desired_levels) {
            return false;
          }
        } else if (gsl::at(neighbor_flags, d) == ::amr::Flag::Join and
                   element_segment.id_of_sibling() == neighbor_segment) {
          return false;
        }
      }
    }
  }
  return true;
}
}  // namespace

namespace TestHelpers::amr {

template <size_t Dim>
valid_flags_t<Dim> valid_neighbor_flags(
    const ElementId<Dim>& element_id,
    const std::array<::amr::Flag, Dim>& element_flags,
    const Neighbors<Dim>& neighbors) {
  valid_flags_t<Dim> result{};
  const auto& orientation_of_neighbors = neighbors.orientation();
  const auto& first_neighbor_id = *(neighbors.begin());
  for (const auto& neighbor_flags : amr_flags<Dim>()) {
    if (are_valid_neighbor_flags(element_id, element_flags, first_neighbor_id,
                                 neighbor_flags, orientation_of_neighbors)) {
      result.emplace_back(
          neighbor_flags_t<Dim>{{first_neighbor_id, neighbor_flags}});
    }
  }

  for (auto it = std::next(neighbors.begin()); it != neighbors.end(); ++it) {
    valid_flags_t<Dim> prev_result{};
    std::swap(result, prev_result);
    const auto& neighbor_id = *it;
    for (const auto& neighbor_flags : amr_flags<Dim>()) {
      if (not are_valid_neighbor_flags(element_id, element_flags, neighbor_id,
                                       neighbor_flags,
                                       orientation_of_neighbors)) {
        // neighbor_flags conflicts with element
        continue;
      }
      for (const auto& valid_flags : prev_result) {
        bool can_add_flag = true;
        for (const auto& [prev_neighbor_id, prev_neighbor_flags] :
             valid_flags) {
          if (not are_valid_neighbor_flags(prev_neighbor_id,
                                           prev_neighbor_flags, neighbor_id,
                                           neighbor_flags)) {
            // neighbor_flags conflicts with previous neighbor
            can_add_flag = false;
            break;
          }
        }
        if (can_add_flag) {
          auto new_flags = valid_flags;
          new_flags.emplace(neighbor_id, neighbor_flags);
          result.emplace_back(std::move(new_flags));
        }
      }
    }
  }
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                   \
  template valid_flags_t<DIM(data)> valid_neighbor_flags(      \
      const ElementId<DIM(data)>& element_id,                  \
      const std::array<::amr::Flag, DIM(data)>& element_flags, \
      const Neighbors<DIM(data)>& neighbors);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
}  // namespace TestHelpers::amr
