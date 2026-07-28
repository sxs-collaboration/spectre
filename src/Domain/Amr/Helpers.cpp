// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Amr/Helpers.hpp"

#include <array>
#include <boost/rational.hpp>
#include <cstddef>
#include <deque>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/StdHelpers.hpp"

namespace amr {
template <size_t VolumeDim>
std::array<size_t, VolumeDim> desired_refinement_levels(
    const ElementId<VolumeDim>& id, const std::array<Flag, VolumeDim>& flags) {
  std::array<size_t, VolumeDim> result = id.refinement_levels();

  for (size_t d = 0; d < VolumeDim; ++d) {
    ASSERT(Flag::Undefined != gsl::at(flags, d),
           "Undefined Flag in dimension " << d);
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
  std::array<size_t, VolumeDim> result =
      orientation.permute_from_neighbor(neighbor_id.refinement_levels());
  for (size_t d = 0; d < VolumeDim; ++d) {
    ASSERT(Flag::Undefined != gsl::at(neighbor_flags, d),
           "Undefined Flag in dimension " << d);
    const size_t mapped_dim = orientation(d);
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
  return {1, two_to_the(alg::accumulate(element_id.refinement_levels(), 0_st))};
}

template <size_t VolumeDim>
bool has_potential_sibling(const ElementId<VolumeDim>& element_id,
                           const Direction<VolumeDim>& direction) {
  const SegmentId segment_id = element_id.segment_id(direction.dimension());
  if (segment_id.refinement_level() == 0) {
    return false;
  }
  return direction.side() == segment_id.side_of_sibling();
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

template <size_t VolumeDim>
std::vector<ElementId<VolumeDim>> ids_of_children(
    const ElementId<VolumeDim>& element_id,
    const std::array<amr::Flag, VolumeDim>& flags,
    const std::optional<size_t>& child_grid_index) {
  using ::operator<<;
  ASSERT(alg::count(flags, Flag::Split) > 0,
         "Element " << element_id << " has no children given flags " << flags);
  ASSERT(alg::count(flags, Flag::Join) == 0,
         "Splitting and joining an Element is not supported");
  const size_t block_id = element_id.block_id();
  const size_t grid_index = child_grid_index.value_or(element_id.grid_index());
  if constexpr (VolumeDim == 1) {
    return {{block_id,
             {{element_id.segment_id(0).id_of_child(Side::Lower)}},
             grid_index},
            {block_id,
             {{element_id.segment_id(0).id_of_child(Side::Upper)}},
             grid_index}};
  } else {
    const auto segment_ids = element_id.segment_ids();
    std::array<std::vector<SegmentId>, VolumeDim> child_segment_ids;
    for (size_t d = 0; d < VolumeDim; ++d) {
      const SegmentId& segment_id = gsl::at(segment_ids, d);
      gsl::at(child_segment_ids, d) =
          gsl::at(flags, d) == Flag::Split
              ? std::vector{segment_id.id_of_child(Side::Lower),
                            segment_id.id_of_child(Side::Upper)}
              : std::vector{segment_id};
    }

    std::vector<ElementId<VolumeDim>> result{};
    for (const auto segment_id_xi : child_segment_ids[0]) {
      for (const auto segment_id_eta : child_segment_ids[1]) {
        if constexpr (VolumeDim == 2) {
          result.emplace_back(
              block_id, std::array{segment_id_xi, segment_id_eta}, grid_index);
        } else {
          for (const auto segment_id_zeta : child_segment_ids[2]) {
            result.emplace_back(
                block_id,
                std::array{segment_id_xi, segment_id_eta, segment_id_zeta},
                grid_index);
          }
        }
      }
    }
    return result;
  }
}

template <size_t VolumeDim>
std::deque<ElementId<VolumeDim>> ids_of_joining_neighbors(
    const Element<VolumeDim>& element,
    const std::array<Flag, VolumeDim>& flags) {
  using ::operator<<;
  ASSERT(alg::count(flags, Flag::Join) > 0,
         "Element " << element.id() << " is not joining given flags " << flags);
  ASSERT(alg::count(flags, Flag::Split) == 0,
         "Splitting and joining an Element is not supported");
  std::deque<ElementId<VolumeDim>> result;
  for (size_t d = 0; d < VolumeDim; ++d) {
    if (gsl::at(flags, d) == Flag::Join) {
      const Direction<VolumeDim> sibling_direction{
          d, element.id().segment_id(d).side_of_sibling()};
      const auto& neighbor_sibling_ids_in_this_dim =
          element.neighbors().at(sibling_direction).ids();
      for (auto sibling_id : neighbor_sibling_ids_in_this_dim) {
        result.emplace_back(sibling_id);
      }
    }
  }
  return result;
}

template <size_t VolumeDim>
bool is_child_that_creates_parent(const ElementId<VolumeDim>& element_id,
                                  const std::array<Flag, VolumeDim>& flags) {
  using ::operator<<;
  ASSERT(alg::count(flags, Flag::Join) > 0,
         "Element " << element_id << " is not joining given flags " << flags);
  ASSERT(alg::count(flags, Flag::Split) == 0,
         "Splitting and joining an Element is not supported");
  for (size_t d = 0; d < VolumeDim; ++d) {
    if (gsl::at(flags, d) == Flag::Join and
        element_id.segment_id(d).index() % 2 == 1) {
      return false;
    }
  }
  return true;
}

template <size_t VolumeDim>
bool prevent_element_from_joining_while_splitting(
    const gsl::not_null<std::array<Flag, VolumeDim>*> flags) {
  bool flags_changed = false;
  if (alg::any_of(*flags,
                  [](amr::Flag flag) { return flag == amr::Flag::Split; })) {
    for (size_t d = 0; d < VolumeDim; ++d) {
      if (gsl::at(*flags, d) == amr::Flag::Join) {
        gsl::at(*flags, d) = amr::Flag::DoNothing;
        flags_changed = true;
      }
    }
  }
  return flags_changed;
}

namespace {
constexpr auto can_be_h_refined =
    std::array{domain::Topology::I1, domain::Topology::B1Radial,
               domain::Topology::B2Radial, domain::Topology::B3Radial};
constexpr auto cannot_be_h_refined =
    std::array{domain::Topology::S1,
               domain::Topology::S2Colatitude,
               domain::Topology::S2Longitude,
               domain::Topology::B2Angular,
               domain::Topology::B3Colatitude,
               domain::Topology::B3Longitude,
               domain::Topology::CartoonSphere,
               domain::Topology::CartoonCylinder};
}  // namespace

template <size_t VolumeDim>
void enforce_h_refinement_topology_restrictions(
    gsl::not_null<std::array<Flag, VolumeDim>*> flags,
    const std::array<domain::Topology, VolumeDim>& topologies) {
  for (size_t d = 0; d < VolumeDim; ++d) {
    auto& flag = gsl::at(*flags, d);
    const auto topology = gsl::at(topologies, d);
    if (flag != amr::Flag::DoNothing) {
      ASSERT(flag == amr::Flag::Split or flag == amr::Flag::Join,
             "Expected h-refinement flag, not " << flag);
      if (alg::found(cannot_be_h_refined, topology)) {
        flag = amr::Flag::DoNothing;
      } else {
        ASSERT(alg::found(can_be_h_refined, topology),
               "Topology " << topology
                           << " not found in either list can_be_h_refined or "
                              "cannot_be_h_refined");
      }
    }
  }
}

template <size_t VolumeDim>
void enforce_p_refinement_topology_restrictions(
    gsl::not_null<std::array<Flag, VolumeDim>*> flags,
    const std::array<domain::Topology, VolumeDim>& topologies) {
  if constexpr (VolumeDim == 2) {
    auto& [first_flag, second_flag] = *flags;
    if ((topologies == domain::topologies::disk or
         topologies == domain::topologies::spherical_surface) and
        first_flag != second_flag) {
      const auto max_flag = std::max(first_flag, second_flag);
      ASSERT(max_flag != amr::Flag::Split and max_flag != amr::Flag::Join,
             "Expected p-refinement flag, not " << max_flag);
      *flags = make_array<2>(max_flag);
    }
  } else if constexpr (VolumeDim == 3) {
    auto& [first_flag, second_flag, third_flag] = *flags;
    if (topologies == domain::topologies::spherical_shell and
        second_flag != third_flag) {
      const auto max_flag = std::max(second_flag, third_flag);
      ASSERT(max_flag != amr::Flag::Split and max_flag != amr::Flag::Join,
             "Expected p-refinement flag, not " << max_flag);
      second_flag = max_flag;
      third_flag = max_flag;
    } else if (topologies == domain::topologies::full_cylinder and
               first_flag != second_flag) {
      const auto max_flag = std::max(first_flag, second_flag);
      ASSERT(max_flag != amr::Flag::Split and max_flag != amr::Flag::Join,
             "Expected p-refinement flag, not " << max_flag);
      first_flag = max_flag;
      second_flag = max_flag;
    } else if (topologies == domain::topologies::full_sphere and
               (first_flag != second_flag or second_flag != third_flag)) {
      const auto max_flag = *(alg::max_element(*flags));
      ASSERT(max_flag != amr::Flag::Split and max_flag != amr::Flag::Join,
             "Expected p-refinement flag, not " << max_flag);
      *flags = make_array<3>(max_flag);
    } else if (topologies[2] == domain::Topology::CartoonSphere) {
      second_flag = amr::Flag::DoNothing;
      third_flag = amr::Flag::DoNothing;
    } else if (topologies[2] == domain::Topology::CartoonCylinder) {
      third_flag = amr::Flag::DoNothing;
    }
  }
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
      const std::array<amr::Flag, DIM(data)>& flags);                          \
  template std::vector<ElementId<DIM(data)>> ids_of_children(                  \
      const ElementId<DIM(data)>& element_id,                                  \
      const std::array<amr::Flag, DIM(data)>& flags,                           \
      const std::optional<size_t>& child_grid_index);                          \
  template std::deque<ElementId<DIM(data)>> ids_of_joining_neighbors(          \
      const Element<DIM(data)>& element,                                       \
      const std::array<Flag, DIM(data)>& flags);                               \
  template bool is_child_that_creates_parent(                                  \
      const ElementId<DIM(data)>& element_id,                                  \
      const std::array<Flag, DIM(data)>& flags);                               \
  template bool prevent_element_from_joining_while_splitting(                  \
      const gsl::not_null<std::array<Flag, DIM(data)>*> flags);                \
  template void enforce_h_refinement_topology_restrictions(                    \
      const gsl::not_null<std::array<Flag, DIM(data)>*> flags,                 \
      const std::array<domain::Topology, DIM(data)>& topologies);              \
  template void enforce_p_refinement_topology_restrictions(                    \
      const gsl::not_null<std::array<Flag, DIM(data)>*> flags,                 \
      const std::array<domain::Topology, DIM(data)>& topologies);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
}  // namespace amr
