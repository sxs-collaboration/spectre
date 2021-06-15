// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/ElementDistribution.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include "Domain/Structure/ElementId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain {

namespace {
// This interleaves the bits of the element index.
// A sketch of a 2D block with 4x2 elements, with bit indices and resulting
// z-curve
//
//        x-->
//        00  01  10  11
// y  0 |  0   2   4   6
// |    |
// v  1 |  1   3   5   7
template <size_t Dim>
size_t z_curve_index(const ElementId<Dim>& element_id) noexcept {
  // for the bit manipulation of the element index, we need to interleave the
  // indices in each dimension in order according to how many bits are in the
  // index representation. This variable stores the refinement level and
  // dimension index in ascending order of refinement level, representing a
  // permutation of the dimensions
  std::array<std::pair<size_t, size_t>, Dim>
      dimension_by_highest_refinement_level;
  for (size_t i = 0; i < Dim; ++i) {
    dimension_by_highest_refinement_level.at(i) =
        std::make_pair(element_id.segment_id(i).refinement_level(), i);
  }
  alg::sort(dimension_by_highest_refinement_level,
            [](const std::pair<size_t, size_t>& lhs,
               const std::pair<size_t, size_t>& rhs) {
              return lhs.first < rhs.first;
            });

  size_t element_order_index = 0;

  // 'gap' the lowest refinement direction bits as:
  // ... x1 x0 -> ... x1 0 0 x0,
  // then bitwise or in 'gap'ed and shifted next-lowest refinement direction
  // bits as:
  // ... y2 y1 y0 -> ... y2 0 y1 x1 0 y0 x0
  // then bitwise or in 'gap'ed and shifted highest-refinement direction bits
  // as:
  // ... z3 z2 z1 z0 -> z3 z2 y2 z1 y1 x1 z0 y0 x0
  // note that we must skip refinement-level 0 dimensions as though they are
  // not present
  size_t leading_gap = 0;
  for (size_t i = 0; i < Dim; ++i) {
    const size_t id_to_gap_and_shift =
        element_id
            .segment_id(
                gsl::at(dimension_by_highest_refinement_level, i).second)
            .index();
    size_t total_gap = leading_gap;
    if (gsl::at(dimension_by_highest_refinement_level, i).first > 0) {
      ++leading_gap;
    }
    for (size_t bit_index = 0;
         bit_index < gsl::at(dimension_by_highest_refinement_level, i).first;
         ++bit_index) {
      // This operation will not overflow for our present use of `ElementId`s.
      // This technique densely assigns an ElementID a unique size_t identifier
      // determining the Morton curve order, and `ElementId` supports refinement
      // levels such that a global index within a block will fit in a 64-bit
      // unsigned integer.
      element_order_index |=
          ((id_to_gap_and_shift & two_to_the(bit_index)) << total_gap);
      for (size_t j = 0; j < Dim; ++j) {
        if (i != j and
            bit_index + 1 <
                gsl::at(dimension_by_highest_refinement_level, j).first) {
          ++total_gap;
        }
      }
    }
  }
  return element_order_index;
}
}  // namespace

template <size_t Dim>
BlockZCurveProcDistribution<Dim>::BlockZCurveProcDistribution(
    size_t number_of_procs,
    const std::vector<std::array<size_t, Dim>>& refinements_by_block) noexcept {
  block_element_distribution_ =
      std::vector<std::vector<std::pair<size_t, size_t>>>(
          refinements_by_block.size());
  auto add_number_of_elements_for_refinement =
      [](size_t lhs, const std::array<size_t, Dim>& rhs) {
        size_t value = 1;
        for (size_t i = 0; i < Dim; ++i) {
          value *= two_to_the(gsl::at(rhs, i));
        }
        return lhs + value;
      };
  const size_t number_of_elements =
      std::accumulate(refinements_by_block.begin(), refinements_by_block.end(),
                      0_st, add_number_of_elements_for_refinement);
  ASSERT(not refinements_by_block.empty(),
         "`refinements_by_block` must be non-empty.");
  // currently, we just assign uniform weight to elements. In future, it will
  // probably be better to take into account p-refinement, but then the z-curve
  // method will also require weighting.
  size_t remaining_elements_in_block =
      add_number_of_elements_for_refinement(0_st, refinements_by_block[0]);
  size_t current_block = 0;
  for (size_t i = 0; i < number_of_procs; ++i) {
    size_t remaining_elements_on_proc =
        (number_of_elements / number_of_procs) +
        (i < (number_of_elements % number_of_procs) ? 1 : 0);
    while (remaining_elements_on_proc > 0) {
      block_element_distribution_.at(current_block)
          .emplace_back(
              std::make_pair(i, std::min(remaining_elements_in_block,
                                         remaining_elements_on_proc)));
      if (remaining_elements_in_block <= remaining_elements_on_proc) {
        remaining_elements_on_proc -= remaining_elements_in_block;
        ++current_block;
        if (current_block < refinements_by_block.size()) {
          remaining_elements_in_block = add_number_of_elements_for_refinement(
              0_st, gsl::at(refinements_by_block, current_block));
        }
      } else {
        remaining_elements_in_block -= remaining_elements_on_proc;
        remaining_elements_on_proc = 0;
      }
    }
  }
}

template <size_t Dim>
size_t BlockZCurveProcDistribution<Dim>::get_proc_for_element(
    const size_t block_id, const ElementId<Dim>& element_id) const noexcept {
  const size_t element_order_index = z_curve_index(element_id);
  size_t total_so_far = 0;
  for (const std::pair<size_t, size_t>& element_info :
       gsl::at(block_element_distribution_, block_id)) {
    if (total_so_far <= element_order_index and
        element_info.second + total_so_far > element_order_index) {
      return element_info.first;
    }
    total_so_far += element_info.second;
  }
  ERROR(
      "Processor not successfully chosen. This indicates a flaw in the logic "
      "of BlockZCurveProcDistribution.");
}
#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class BlockZCurveProcDistribution<GET_DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace domain
