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
#include "Domain/Structure/ZCurve.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain {
template <size_t Dim>
BlockZCurveProcDistribution<Dim>::BlockZCurveProcDistribution(
    size_t number_of_procs_with_elements,
    const std::vector<std::array<size_t, Dim>>& refinements_by_block,
    const std::unordered_set<size_t>& global_procs_to_ignore) {
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
  // This variable will keep track of how many global procs we've skipped over
  // so far. This bookkeeping is necessary so the element gets placed on the
  // correct global proc. The loop variable `i` does not correspond to global
  // proc number. It's just an index
  size_t number_of_ignored_procs_so_far = 0;
  for (size_t i = 0; i < number_of_procs_with_elements; ++i) {
    size_t global_proc_number = i + number_of_ignored_procs_so_far;
    while (global_procs_to_ignore.find(global_proc_number) !=
           global_procs_to_ignore.end()) {
      ++number_of_ignored_procs_so_far;
      ++global_proc_number;
    }
    size_t remaining_elements_on_proc =
        (number_of_elements / number_of_procs_with_elements) +
        (i < (number_of_elements % number_of_procs_with_elements) ? 1 : 0);
    while (remaining_elements_on_proc > 0) {
      block_element_distribution_.at(current_block)
          .emplace_back(std::make_pair(global_proc_number,
                                       std::min(remaining_elements_in_block,
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
    const ElementId<Dim>& element_id) const {
  const size_t element_order_index = z_curve_index(element_id);
  size_t total_so_far = 0;
  for (const std::pair<size_t, size_t>& element_info :
       gsl::at(block_element_distribution_, element_id.block_id())) {
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
