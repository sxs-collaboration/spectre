// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/LinearSolver/Multigrid/Hierarchy.hpp"

#include <array>
#include <cstddef>
#include <unordered_set>
#include <vector>

#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace LinearSolver::multigrid {

template <size_t Dim>
std::vector<std::array<size_t, Dim>> coarsen(
    std::vector<std::array<size_t, Dim>> initial_refinement_levels) noexcept {
  for (auto& block_refinement : initial_refinement_levels) {
    for (size_t d = 0; d < Dim; ++d) {
      auto& refinement_level = gsl::at(block_refinement, d);
      if (refinement_level > 0) {
        --refinement_level;
      }
    }
  }
  return initial_refinement_levels;
}

template <size_t Dim>
ElementId<Dim> parent_id(const ElementId<Dim>& child_id) noexcept {
  std::array<SegmentId, Dim> parent_segment_ids = child_id.segment_ids();
  for (size_t d = 0; d < Dim; ++d) {
    auto& segment_id = gsl::at(parent_segment_ids, d);
    if (segment_id.refinement_level() > 0) {
      segment_id = segment_id.id_of_parent();
    }
  }
  return {child_id.block_id(), std::move(parent_segment_ids),
          child_id.grid_index() + 1};
}

namespace {
template <size_t Dim>
void assert_finest_grid(
    const std::array<SegmentId, Dim>& parent_segment_ids,
    const std::array<size_t, Dim>& children_refinement_levels) noexcept {
  for (size_t d = 0; d < Dim; ++d) {
    ASSERT(gsl::at(children_refinement_levels, d) ==
               gsl::at(parent_segment_ids, d).refinement_level(),
           "On the finest grid, expected the children refinement levels "
               << children_refinement_levels
               << " to equal the refinement levels of the parent segment IDs "
               << parent_segment_ids << " in dimension " << d << ".");
  }
}

std::unordered_set<SegmentId> child_segment_ids_impl(
    const SegmentId& parent_segment_id,
    const size_t children_refinement_level) noexcept {
  ASSERT(parent_segment_id.refinement_level() == children_refinement_level or
             (children_refinement_level > 0 and
              parent_segment_id.refinement_level() ==
                  children_refinement_level - 1),
         "The parent refinement level must be exactly 1 smaller than the "
         "children refinement level, or the same. Parent refinement level: "
             << parent_segment_id.refinement_level()
             << ", children refinement level: " << children_refinement_level);
  if (parent_segment_id.refinement_level() < children_refinement_level) {
    return {parent_segment_id.id_of_child(Side::Lower),
            parent_segment_id.id_of_child(Side::Upper)};
  } else {
    return {parent_segment_id};
  }
}
}  // namespace

template <>
std::unordered_set<ElementId<1>> child_ids<1>(
    const ElementId<1>& parent_id,
    const std::array<size_t, 1>& children_refinement_levels) noexcept {
  if (parent_id.grid_index() == 0) {
#ifdef SPECTRE_DEBUG
    assert_finest_grid(parent_id.segment_ids(), children_refinement_levels);
#endif  // SPECTRE_DEBUG
    return {};
  }
  const std::unordered_set<SegmentId> child_segment_ids =
      child_segment_ids_impl(parent_id.segment_ids()[0],
                             children_refinement_levels[0]);
  std::unordered_set<ElementId<1>> child_ids{};
  for (const auto& child_segment_id : child_segment_ids) {
    child_ids.emplace(parent_id.block_id(),
                      std::array<SegmentId, 1>{child_segment_id},
                      parent_id.grid_index() - 1);
  }
  return child_ids;
}

template <>
std::unordered_set<ElementId<2>> child_ids<2>(
    const ElementId<2>& parent_id,
    const std::array<size_t, 2>& children_refinement_levels) noexcept {
  if (parent_id.grid_index() == 0) {
#ifdef SPECTRE_DEBUG
    assert_finest_grid(parent_id.segment_ids(), children_refinement_levels);
#endif  // SPECTRE_DEBUG
    return {};
  }
  std::array<std::unordered_set<SegmentId>, 2> child_segment_ids{};
  for (size_t d = 0; d < 2; ++d) {
    gsl::at(child_segment_ids, d) =
        child_segment_ids_impl(gsl::at(parent_id.segment_ids(), d),
                               gsl::at(children_refinement_levels, d));
  }
  std::unordered_set<ElementId<2>> child_ids{};
  for (const auto& child_segment_id_x : child_segment_ids[0]) {
    for (const auto& child_segment_id_y : child_segment_ids[1]) {
      child_ids.emplace(
          parent_id.block_id(),
          std::array<SegmentId, 2>{{child_segment_id_x, child_segment_id_y}},
          parent_id.grid_index() - 1);
    }
  }
  return child_ids;
}

template <>
std::unordered_set<ElementId<3>> child_ids<3>(
    const ElementId<3>& parent_id,
    const std::array<size_t, 3>& children_refinement_levels) noexcept {
  if (parent_id.grid_index() == 0) {
#ifdef SPECTRE_DEBUG
    assert_finest_grid(parent_id.segment_ids(), children_refinement_levels);
#endif  // SPECTRE_DEBUG
    return {};
  }
  std::array<std::unordered_set<SegmentId>, 3> child_segment_ids{};
  for (size_t d = 0; d < 3; ++d) {
    gsl::at(child_segment_ids, d) =
        child_segment_ids_impl(gsl::at(parent_id.segment_ids(), d),
                               gsl::at(children_refinement_levels, d));
  }
  std::unordered_set<ElementId<3>> child_ids{};
  for (const auto& child_segment_id_x : child_segment_ids[0]) {
    for (const auto& child_segment_id_y : child_segment_ids[1]) {
      for (const auto& child_segment_id_z : child_segment_ids[2]) {
        child_ids.emplace(
            parent_id.block_id(),
            std::array<SegmentId, 3>{
                {child_segment_id_x, child_segment_id_y, child_segment_id_z}},
            parent_id.grid_index() - 1);
      }
    }
  }
  return child_ids;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(r, data)                                   \
  template std::vector<std::array<size_t, DIM(data)>> coarsen( \
      std::vector<std::array<size_t, DIM(data)>>               \
          initial_refinement_levels) noexcept;                 \
  template ElementId<DIM(data)> parent_id(                     \
      const ElementId<DIM(data)>& child_id) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace LinearSolver::multigrid
