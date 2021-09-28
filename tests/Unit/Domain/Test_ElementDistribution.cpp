// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <set>
#include <vector>

#include "Domain/ElementDistribution.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Utilities/Literals.hpp"

namespace {

// a global indexing to help aggregating the processor maps in the tests
template <size_t Dim>
std::array<SegmentId, Dim> element_index_to_segment_id(
    const std::array<size_t, Dim>& refinement_levels,
    const size_t element_index) {
  std::array<SegmentId, Dim> segment_ids{};
  size_t stride = 1;
  for (size_t i = 0; i < Dim; ++i) {
    const size_t element_index_in_dim =
        (element_index / stride) % two_to_the(gsl::at(refinement_levels, i));
    segment_ids.at(i) =
        SegmentId{gsl::at(refinement_levels, i), element_index_in_dim};
    stride *= two_to_the(gsl::at(refinement_levels, i));
  }
  return segment_ids;
}

template <size_t Dim>
size_t segment_id_to_element_index(
    const std::array<SegmentId, Dim>& segment_ids) {
  size_t element_index = 0;
  size_t stride = 1;
  for (size_t i = 0; i < Dim; ++i) {
    element_index += stride * gsl::at(segment_ids, i).index();
    stride *= two_to_the(gsl::at(segment_ids, i).refinement_level());
  }
  return element_index;
}

template <size_t Dim>
size_t number_of_elements_in_block(
    const std::array<size_t, Dim>& refinement_levels) {
  size_t number_of_elements = 1;
  for (size_t i = 0; i < Dim; ++i) {
    number_of_elements *= two_to_the(gsl::at(refinement_levels, i));
  }
  return number_of_elements;
}

template <size_t Dim>
std::vector<std::vector<size_t>> make_proc_map_for_domain(
    const size_t number_of_blocks, const size_t number_of_procs,
    const std::vector<std::array<size_t, Dim>>& refinement_levels_by_block) {
  std::vector<std::vector<size_t>> proc_map(number_of_blocks);
  const domain::BlockZCurveProcDistribution distribution{
      number_of_procs, refinement_levels_by_block};
  for (size_t block = 0; block < number_of_blocks; ++block) {
    const size_t number_of_elements =
        number_of_elements_in_block(gsl::at(refinement_levels_by_block, block));
    proc_map.at(block) = std::vector<size_t>(number_of_elements);
    for (size_t element_index = 0; element_index < number_of_elements;
         ++element_index) {
      const ElementId<Dim> element_id{
          block,
          element_index_to_segment_id(
              gsl::at(refinement_levels_by_block, block), element_index)};
      proc_map.at(block).at(element_index) =
          distribution.get_proc_for_element(element_id);
    }
  }
  return proc_map;
}

// check that the distribution portions out the number of elements approximately
// evenly (within 1 element) to each processor
template <size_t Dim>
void check_element_distribution_uniformity(
    const std::vector<std::vector<size_t>>& proc_map,
    const size_t number_of_procs,
    const std::vector<std::array<size_t, Dim>>& refinement_levels_by_block) {
  std::vector<size_t> elements_per_proc(number_of_procs);
  for (const auto& block_proc_map : proc_map) {
    for (const size_t proc : block_proc_map) {
      ++elements_per_proc.at(proc);
    }
  }
  const size_t number_of_elements = std::accumulate(
      refinement_levels_by_block.begin(), refinement_levels_by_block.end(),
      0_st, [](const size_t lhs, const std::array<size_t, Dim>& rhs) {
        return lhs + number_of_elements_in_block(rhs);
      });
  for (const size_t element_count : elements_per_proc) {
    CHECK((element_count == number_of_elements / number_of_procs or
           element_count == number_of_elements / number_of_procs + 1));
  }
}

// check that, for each block, no processor appears in more than two connected
// groups, and that each processor does not appear in too many blocks.
template <size_t Dim>
void check_element_distribution_cohesion(
    const std::vector<std::vector<size_t>>& proc_map,
    const size_t number_of_procs,
    const std::vector<std::array<size_t, Dim>>& refinement_levels_by_block,
    const bool nonuniform_block = false) {
  std::vector<std::set<size_t>> block_set_per_proc(number_of_procs);
  for (size_t block = 0; block < proc_map.size(); ++block) {
    std::array<size_t, Dim> strides{};
    strides[0] = 1;
    for (size_t i = 1; i < Dim; ++i) {
      strides.at(i) = gsl::at(strides, i - 1) *
                      two_to_the(gsl::at(
                          gsl::at(refinement_levels_by_block, block), i - 1));
    }
    std::vector<size_t> number_of_clusters_per_proc(number_of_procs, 0_st);
    std::vector<bool> seen(gsl::at(proc_map, block).size(), false);
    for (size_t start_element = 0;
         start_element < gsl::at(proc_map, block).size(); ++start_element) {
      if (not seen.at(start_element)) {
        const size_t current_proc =
            gsl::at(gsl::at(proc_map, block), start_element);
        block_set_per_proc.at(current_proc).insert(block);
        ++number_of_clusters_per_proc.at(current_proc);
        seen.at(start_element) = true;
        // perform a bredth-first search to get all elements in the cluster that
        // share a proc and mark them seen
        std::deque<size_t> next_elements;
        auto insert_adjacent_unseen = [&next_elements, &current_proc, &proc_map,
                                       &refinement_levels_by_block, &strides,
                                       &seen, &block](const size_t index) {
          for (size_t i = 0; i < Dim; ++i) {
            const size_t index_in_dim =
                (index / gsl::at(strides, i)) %
                two_to_the(
                    gsl::at(gsl::at(refinement_levels_by_block, block), i));
            if (index_in_dim > 0 and
                not seen.at(index - gsl::at(strides, i)) and
                gsl::at(gsl::at(proc_map, block),
                        index - gsl::at(strides, i)) == current_proc) {
              next_elements.push_back(index - gsl::at(strides, i));
              seen.at(index - gsl::at(strides, i)) = true;
            }
            if (index_in_dim + 1 <
                    two_to_the(gsl::at(
                        gsl::at(refinement_levels_by_block, block), i)) and
                not seen.at(index + gsl::at(strides, i)) and
                gsl::at(gsl::at(proc_map, block),
                        index + gsl::at(strides, i)) == current_proc) {
              next_elements.push_back(index + gsl::at(strides, i));
              seen.at(index + gsl::at(strides, i)) = true;
            }
          }
        };
        insert_adjacent_unseen(start_element);
        // note that this loop cannot be converted to an STL iterator algorithm
        // because `insert_adjacent_unseen` violates iterator stability via the
        // insertions.
        while (not next_elements.empty()) {
          const size_t front = next_elements.front();
          next_elements.pop_front();
          insert_adjacent_unseen(front);
        }
      }
    }
    // verify that the distribution is well-clustered -- the Z-curve should
    // ensure no more than 2 clusters for each core
    for (const size_t number_of_clusters : number_of_clusters_per_proc) {
      CHECK(number_of_clusters < 3);
    }
  }
  // verify that each processor has not been assigned to too many blocks -- the
  // greedy algorithm will let extras 'overflow' into the next block, but that
  // shouldn't give more than one extra block in the count.
  // this check assumes that only one block is a different size to make the
  // upper bound calculation simple -- the algorithm still works to keep the
  // number of blocks in which a given processor participates low for more
  // intricate cases, but the upper bound becomes more complicated to write out.
  for (const auto& blocks : block_set_per_proc) {
    CHECK(blocks.size() <=
          static_cast<size_t>(
              std::ceil(static_cast<double>(refinement_levels_by_block.size()) /
                        number_of_procs) +
              (nonuniform_block ? 2 : 1)));
  }
}

template <size_t Dim>
void test_single_block_domain() {
  std::vector<std::array<size_t, Dim>> refinement_levels_by_block;
  refinement_levels_by_block.emplace_back();
  for (size_t i = 0; i < Dim; ++i) {
    refinement_levels_by_block.at(0).at(i) = 3;
  }

  std::vector<std::vector<size_t>> proc_map =
      make_proc_map_for_domain(1, two_to_the(Dim), refinement_levels_by_block);
  // check that the domain is segmented into 4x4x4 cubes
  std::set<size_t> procs_seen;
  for (size_t cube_index = 0; cube_index < two_to_the(Dim); ++cube_index) {
    std::array<SegmentId, Dim> reference_segment_id{};
    size_t stride = 1;
    for (size_t i = 0; i < Dim; ++i) {
      reference_segment_id.at(i) =
          SegmentId{gsl::at(gsl::at(refinement_levels_by_block, 0), i),
                    (cube_index / stride) % 2 == 0_st ? 0_st : 4_st};
      stride *= 2;
    }
    const size_t global_reference_id =
        segment_id_to_element_index(reference_segment_id);
    // check that the 2^Dim lower corners of the large cubes have unique procs
    CHECK(procs_seen.count(
              gsl::at(gsl::at(proc_map, 0), global_reference_id)) == 0);
    procs_seen.insert(gsl::at(gsl::at(proc_map, 0), global_reference_id));
    for (size_t element_within_cube = 0;
         element_within_cube <
         number_of_elements_in_block(gsl::at(refinement_levels_by_block, 0)) /
             two_to_the(Dim);
         ++element_within_cube) {
      std::array<SegmentId, Dim> segment_id = reference_segment_id;
      stride = 1;
      for (size_t i = 0; i < Dim; ++i) {
        segment_id.at(i) = SegmentId{gsl::at(segment_id, i).refinement_level(),
                                     gsl::at(segment_id, i).index() +
                                         (element_within_cube / stride) % 4};
        stride *= 4_st;
      }
      // check that each element in the large cubes have the same processor as
      // the corner elements.
      size_t global_element_id = segment_id_to_element_index(segment_id);
      CHECK(gsl::at(gsl::at(proc_map, 0), global_element_id) ==
            gsl::at(gsl::at(proc_map, 0), global_reference_id));
    }
  }

  // test a more scattered distribution because of the prime number of procs
  proc_map = make_proc_map_for_domain(1, 5, refinement_levels_by_block);
  check_element_distribution_uniformity(proc_map, 5,
                                        refinement_levels_by_block);
  check_element_distribution_cohesion(proc_map, 5, refinement_levels_by_block);
}

template <size_t Dim>
void general_test(const size_t number_of_blocks, const size_t number_of_procs,
                  const bool uneven_domain) {
  std::vector<std::array<size_t, Dim>> refinement_levels_by_block;
  for (size_t i = 0; i < number_of_blocks; ++i) {
    refinement_levels_by_block.emplace_back();
    for (size_t j = 0; j < Dim; ++j) {
      refinement_levels_by_block.at(i).at(j) = uneven_domain ? j : 1;
    }
  }
  if (uneven_domain) {
    for (size_t j = 0; j < Dim; ++j) {
      refinement_levels_by_block.at(0).at(j) = j % 2 == 0 ? 1 : 2;
    }
  }
  const std::vector<std::vector<size_t>> proc_map = make_proc_map_for_domain(
      number_of_blocks, number_of_procs, refinement_levels_by_block);
  check_element_distribution_uniformity(proc_map, number_of_procs,
                                        refinement_levels_by_block);
  check_element_distribution_cohesion(
      proc_map, number_of_procs, refinement_levels_by_block, uneven_domain);
}

SPECTRE_TEST_CASE("Unit.Domain.ElementDistribution", "[Domain][Unit]") {
  {
    INFO("Single block domain");
    test_single_block_domain<1>();
    test_single_block_domain<2>();
    test_single_block_domain<3>();
  }
  for (const size_t number_of_blocks : {2_st, 6_st}) {
    for (const size_t number_of_procs : {2_st, 7_st}) {
      for (const bool uneven_domain : {true, false}) {
        general_test<1>(number_of_blocks, number_of_procs, uneven_domain);
        general_test<2>(number_of_blocks, number_of_procs, uneven_domain);
        general_test<3>(number_of_blocks, number_of_procs, uneven_domain);
      }
    }
  }
}
}  // namespace
