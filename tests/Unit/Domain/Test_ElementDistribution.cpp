// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/ZCurve.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace {
// Test the weighting done by `domain::get_element_costs` for a uniform cost
// function
void test_uniform_cost_function() {
  // AlignedLattice with differently-refined Elements
  const auto domain_creator = domain::creators::AlignedLattice<2>(
      {{{{70, 71, 72, 73}}, {{90, 92, 95, 99}}}}, {{2, 5}}, {{3, 3}},
      {{{{{1, 0}}, {{3, 2}}, {{3, 5}}}, {{{2, 1}}, {{3, 3}}, {{4, 6}}}}},
      {{{{{1, 0}}, {{3, 2}}, {{4, 5}}}, {{{2, 1}}, {{3, 3}}, {{6, 7}}}}}, {});

  const auto domain = domain_creator.create_domain();
  const auto& blocks = domain.blocks();

  const auto costs = domain::get_element_costs(
      blocks, domain_creator.initial_refinement_levels(),
      domain_creator.initial_extents(), domain::ElementWeight::Uniform,
      std::nullopt);

  for (const auto& element_id_and_cost : costs) {
    CHECK(element_id_and_cost.second == 1.0);
  }
}

// Test the weighting done by `domain::get_element_costs` for weighted cost
// functions
void test_weighted_cost_function(const domain::ElementWeight element_weight) {
  const auto domain_creator1 = domain::creators::AlignedLattice<3>(
      {{{{0.0, 1.0, 2.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}}}, {{2, 1, 0}},
      {{4, 4, 4}}, {}, {}, {});
  const auto domain1 = domain_creator1.create_domain();
  const auto& blocks1 = domain1.blocks();

  // Block size and grid points are the same as blocks in `domain1`, but
  // refinement levels are different
  const auto domain_creator2 = domain::creators::AlignedLattice<3>(
      {{{{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}}}, {{2, 3, 2}}, {{4, 4, 4}},
      {}, {}, {});
  const auto domain2 = domain_creator2.create_domain();
  const auto& blocks2 = domain2.blocks();

  // Block size and refinement levels are the same as blocks in `domain1`, but
  // grid points are different
  const auto domain_creator3 = domain::creators::AlignedLattice<3>(
      {{{{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}}}, {{2, 1, 0}}, {{4, 3, 2}},
      {}, {}, {});
  const auto domain3 = domain_creator3.create_domain();
  const auto& blocks3 = domain2.blocks();

  const auto costs1 = domain::get_element_costs(
      blocks1, domain_creator1.initial_refinement_levels(),
      domain_creator1.initial_extents(), element_weight,
      Spectral::Quadrature::GaussLobatto);

  const auto costs2 = domain::get_element_costs(
      blocks2, domain_creator2.initial_refinement_levels(),
      domain_creator2.initial_extents(), element_weight,
      Spectral::Quadrature::GaussLobatto);

  const auto costs3 = domain::get_element_costs(
      blocks3, domain_creator3.initial_refinement_levels(),
      domain_creator3.initial_extents(), element_weight,
      Spectral::Quadrature::GaussLobatto);

  // check that all elements in each domain have the same cost

  auto elemental_cost_it1 = costs1.begin();
  const double elemental_cost1 = elemental_cost_it1->second;
  elemental_cost_it1++;
  while (elemental_cost_it1 != costs1.end()) {
    CHECK(elemental_cost1 == approx(elemental_cost_it1->second));
    elemental_cost_it1++;
  }

  auto elemental_cost_it2 = costs2.begin();
  const double elemental_cost2 = elemental_cost_it2->second;
  elemental_cost_it2++;
  while (elemental_cost_it2 != costs1.end()) {
    CHECK(elemental_cost2 == approx(elemental_cost_it2->second));
    elemental_cost_it2++;
  }

  auto elemental_cost_it3 = costs3.begin();
  const double elemental_cost3 = elemental_cost_it3->second;
  elemental_cost_it3++;
  while (elemental_cost_it3 != costs3.end()) {
    CHECK(elemental_cost3 == approx(elemental_cost_it3->second));
    elemental_cost_it3++;
  }

  if (element_weight == domain::ElementWeight::NumGridPoints) {
    // check that varying refinement doesn't affect the cost
    CHECK(elemental_cost2 == elemental_cost1);
  } else {
    // element_weight == domain::ElementWeight::NumGridPointsAndGridSpacing

    // The highest refinement for the first test domain is 2 while the highest
    // refinement for the second test domain is 3, grid points held constant.
    // Since the minimum grid spacing of the second domain is half the minimum
    // grid spacing of the first and since
    // elemental cost = (# of grid points) / sqrt(min grid spacing), the
    // elemental cost of the second domain should be a factor of sqrt(2) the
    // cost.
    CHECK(elemental_cost2 == approx(sqrt(2.0) * elemental_cost1));
  }

  // The minimum grid spacing for the first and third domain are equal, but the
  // number of grid points in an element in the first is 64 while the number of
  // grid points in an element in the third is 24. Since elemental cost for
  // either domain::Elementweight::NumGridPoints or
  // domain::Elementweight::NumGridPointsAndGridSpacing should only scale by
  // the # of grid points, the elemental cost of the third domain should be a
  // factor of 24/64 = 3/8 the cost of the first domain.
  CHECK(elemental_cost3 == elemental_cost1 * 3.0 / 8.0);
}

// Test the processor distribution logic of the
// `domain::BlockZCurveProcDistribution` constructor for an unweighted element
// distribution
template <size_t Dim>
void test_uniform_element_distribution_construction(
    const DomainCreator<Dim>& domain_creator,
    const size_t number_of_procs_with_elements,
    const std::unordered_set<size_t>& global_procs_to_ignore = {}) {
  const auto domain = domain_creator.create_domain();
  const auto& blocks = domain.blocks();
  const auto initial_refinement_levels =
      domain_creator.initial_refinement_levels();
  const auto initial_extents = domain_creator.initial_extents();

  const size_t num_blocks = blocks.size();
  size_t num_elements = 0;
  std::vector<size_t> num_elements_by_block(num_blocks, 0);
  for (size_t i = 0; i < num_blocks; i++) {
    size_t num_elements_this_block =
        two_to_the(gsl::at(initial_refinement_levels[i], 0));
    for (size_t j = 1; j < Dim; j++) {
      num_elements_this_block *=
          two_to_the(gsl::at(initial_refinement_levels[i], j));
    }
    num_elements_by_block[i] = num_elements_this_block;
    num_elements += num_elements_this_block;
  }

  const auto costs = domain::get_element_costs(
      blocks, initial_refinement_levels, initial_extents,
      domain::ElementWeight::Uniform, std::nullopt);

  const domain::BlockZCurveProcDistribution<Dim> element_distribution(
      costs, number_of_procs_with_elements, blocks, initial_refinement_levels,
      initial_extents, global_procs_to_ignore);
  const auto proc_map = element_distribution.block_element_distribution();

  const size_t total_procs =
      number_of_procs_with_elements + global_procs_to_ignore.size();
  std::vector<size_t> num_elements_by_proc(total_procs, 0);
  std::vector<size_t> actual_num_elements_by_block_in_dist(num_blocks, 0);
  size_t actual_num_elements_in_dist = 0;
  for (size_t block_number = 0; block_number < proc_map.size();
       block_number++) {
    for (const auto& proc_allowance : proc_map[block_number]) {
      const size_t proc_number = proc_allowance.first;
      const size_t element_allowance = proc_allowance.second;
      num_elements_by_proc[proc_number] += element_allowance;
      actual_num_elements_by_block_in_dist[block_number] += element_allowance;
    }
    // check that the number of elements in this block accounted for in the
    // distribution matches the expected number of total elements for this block
    CHECK(actual_num_elements_by_block_in_dist[block_number] ==
          num_elements_by_block[block_number]);
    actual_num_elements_in_dist +=
        actual_num_elements_by_block_in_dist[block_number];
  }
  // check that the number of elements accounted for in the distribution matches
  // the expected number of total elements
  CHECK(actual_num_elements_in_dist == num_elements);

  size_t lowest_proc_with_elements = 0;
  while (global_procs_to_ignore.count(lowest_proc_with_elements) == 1) {
    lowest_proc_with_elements++;
  }
  const size_t num_elements_on_lowest_proc =
      num_elements_by_proc[lowest_proc_with_elements];

  size_t num_elements_so_far = num_elements_on_lowest_proc;
  size_t proc_num = lowest_proc_with_elements + 1;
  while (proc_num < total_procs and num_elements_so_far < num_elements) {
    const size_t num_elements_this_proc = num_elements_by_proc[proc_num];
    if (global_procs_to_ignore.count(proc_num) == 1) {
      CHECK(num_elements_this_proc == 0);
    } else {
      // check that the distribution is near-uniform
      CHECK((num_elements_this_proc == num_elements_on_lowest_proc or
             num_elements_this_proc == num_elements_on_lowest_proc + 1 or
             num_elements_this_proc == num_elements_on_lowest_proc - 1));
    }
    num_elements_so_far += num_elements_this_proc;
    proc_num++;
  }

  // check that any remainder of processors we didn't need do indeed have 0
  // elements assigned to them
  while (proc_num < total_procs) {
    CHECK(num_elements_by_proc[proc_num] == 0);
    proc_num++;
  }
}

// Test the processor distribution logic of the
// `domain::BlockZCurveProcDistribution` constructor for weighted element
// distributions
template <size_t Dim>
void test_weighted_element_distribution_construction(
    const domain::ElementWeight element_weight,
    const DomainCreator<Dim>& domain_creator,
    const size_t number_of_procs_with_elements,
    const std::unordered_set<size_t>& global_procs_to_ignore = {}) {
  const auto domain = domain_creator.create_domain();
  const auto& blocks = domain.blocks();
  const auto initial_refinement_levels =
      domain_creator.initial_refinement_levels();
  const auto initial_extents = domain_creator.initial_extents();

  const size_t num_blocks = blocks.size();
  size_t num_elements = 0;
  std::vector<size_t> num_elements_by_block(num_blocks, 0);
  for (size_t i = 0; i < num_blocks; i++) {
    size_t num_elements_this_block =
        two_to_the(gsl::at(initial_refinement_levels[i], 0));
    for (size_t j = 1; j < Dim; j++) {
      num_elements_this_block *=
          two_to_the(gsl::at(initial_refinement_levels[i], j));
    }
    num_elements_by_block[i] = num_elements_this_block;
    num_elements += num_elements_this_block;
  }

  const auto costs = domain::get_element_costs(
      blocks, initial_refinement_levels, initial_extents, element_weight,
      Spectral::Quadrature::GaussLobatto);

  const domain::BlockZCurveProcDistribution<Dim> element_distribution(
      costs, number_of_procs_with_elements, blocks, initial_refinement_levels,
      initial_extents, global_procs_to_ignore);
  const auto proc_map = element_distribution.block_element_distribution();

  const size_t total_procs =
      number_of_procs_with_elements + global_procs_to_ignore.size();
  std::vector<size_t> num_elements_by_proc(total_procs, 0);
  std::vector<size_t> actual_num_elements_by_block_in_dist(num_blocks, 0);
  size_t actual_num_elements_in_dist = 0;
  for (size_t block_number = 0; block_number < proc_map.size();
       block_number++) {
    for (const auto& proc_allowance : proc_map[block_number]) {
      const size_t proc_number = proc_allowance.first;
      const size_t element_allowance = proc_allowance.second;
      num_elements_by_proc[proc_number] += element_allowance;
      actual_num_elements_by_block_in_dist[block_number] += element_allowance;
    }
    // check that the number of elements in this block accounted for in the
    // distribution matches the expected number of total elements for this block
    CHECK(actual_num_elements_by_block_in_dist[block_number] ==
          num_elements_by_block[block_number]);
    actual_num_elements_in_dist +=
        actual_num_elements_by_block_in_dist[block_number];
  }
  // check that the number of elements accounted for in the distribution matches
  // the expected number of total elements
  CHECK(actual_num_elements_in_dist == num_elements);

  std::vector<std::vector<ElementId<Dim>>> initial_element_ids_by_block(
      num_blocks);
  for (size_t i = 0; i < num_blocks; i++) {
    const size_t num_elements_this_block = two_to_the(alg::accumulate(
        initial_refinement_levels[i], 0_st, std::plus<size_t>()));
    initial_element_ids_by_block[i].reserve(num_elements_this_block);
    initial_element_ids_by_block[i] =
        initial_element_ids(blocks[i].id(), initial_refinement_levels[i]);
    alg::sort(initial_element_ids_by_block[i],
              [](const ElementId<Dim>& lhs, const ElementId<Dim>& rhs) {
                return domain::z_curve_index(lhs) < domain::z_curve_index(rhs);
              });
  }

  double total_cost = 0.0;
  for (const auto& element_id_and_cost : costs) {
    total_cost += element_id_and_cost.second;
  }

  // one flattened vector instead of vectors by Block
  std::vector<double> costs_flattened(num_elements);
  size_t cost_index = 0;
  for (size_t i = 0; i < num_blocks; i++) {
    const size_t num_elements_this_block =
        initial_element_ids_by_block[i].size();
    for (size_t j = 0; j < num_elements_this_block; j++) {
      const ElementId<Dim>& element_id = initial_element_ids_by_block[i][j];
      costs_flattened[cost_index] = costs.at(element_id);
      cost_index++;
    }
  }

  cost_index = 0;
  double cost_remaining = total_cost;
  size_t procs_skipped = 0;
  size_t proc_num = 0;
  // check that we distributed the right number of elements to each proc based
  // on the sum of their costs in Z-curve index order
  while (proc_num < total_procs) {
    if (global_procs_to_ignore.count(proc_num)) {
      procs_skipped++;
      proc_num++;
      continue;
    }

    if (cost_index < num_elements) {
      // if we haven't accounted for all elements yet, we should still have cost
      // left to account for
      CHECK(cost_remaining <= total_cost);
    } else {
      // if we've already accounted for all the elements, we shouldn't have any
      // cost left to account for, and it's the case that more procs were
      // requested than could be used, e.g. in the case of less elements than
      // procs
      Approx custom_approx = Approx::custom().epsilon(1.0e-9).scale(1.0);
      CHECK(cost_remaining == custom_approx(0.0));
      break;
    }

    // the average cost per proc that we're aiming for
    const double target_proc_cost =
        cost_remaining /
        (number_of_procs_with_elements - proc_num + procs_skipped);

    // total cost on the processor before adding the cost of the final element
    // assigned to this proc
    double proc_cost_without_final_element = 0.0;
    const size_t num_elements_this_proc = num_elements_by_proc[proc_num];
    // add up costs of all elements but the final one to add
    for (size_t j = 0; j < num_elements_this_proc - 1; j++) {
      const double this_cost = costs_flattened[cost_index + j];
      proc_cost_without_final_element += this_cost;
    }

    // the cost of all of the elements assigned to this proc
    const double proc_cost_with_final_element =
        proc_cost_without_final_element +
        costs_flattened[cost_index + num_elements_this_proc - 1];

    const double diff_without_final_element =
        abs(proc_cost_without_final_element - target_proc_cost);

    const double diff_with_final_element =
        abs(proc_cost_with_final_element - target_proc_cost);

    // if the elements assigned to this proc have a cost that is over the target
    // cost per proc, make sure that either it's because only one element is
    // being assigned to the proc or this cost is closer to the target than if
    // we omitted the final element, i.e. check that it's better to keep the
    // final element than to not
    if (proc_cost_with_final_element > target_proc_cost) {
      CHECK((num_elements_this_proc == 1 or
             diff_with_final_element == approx(diff_without_final_element) or
             diff_with_final_element < diff_without_final_element));
    }

    if (cost_index + num_elements_this_proc < num_elements) {
      // total cost on the processor if we were to add the cost of the next
      // element (one additional than the number chosen)
      const double proc_cost_with_extra_element =
          proc_cost_with_final_element +
          costs_flattened[cost_index + num_elements_this_proc];
      const double diff_with_extra_element =
          abs(proc_cost_with_extra_element - target_proc_cost);

      // if it appears better to add one more element, check that it's because
      // the distance from the target cost with or without the additional
      // element is about the same
      if (diff_with_extra_element < diff_with_final_element) {
        Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
        CHECK(diff_with_extra_element ==
              custom_approx(diff_with_final_element));
      }
    }

    cost_index += num_elements_this_proc;
    cost_remaining -= proc_cost_with_final_element;
    proc_num++;
  }

  // check that any remainder of processors we didn't need do indeed have 0
  // elements assigned to them
  while (proc_num < total_procs) {
    CHECK(num_elements_by_proc[proc_num] == 0);
    proc_num++;
  }
}

// Test the retrieval of the assigned processor that is done by
// `domain::BlockZCurveProcDistribution::get_proc_for_element`
template <size_t Dim>
void test_proc_retrieval(
    const domain::ElementWeight element_weight,
    const DomainCreator<Dim>& domain_creator,
    const size_t number_of_procs_with_elements,
    const std::unordered_set<size_t>& global_procs_to_ignore = {}) {
  const auto domain = domain_creator.create_domain();
  const auto& blocks = domain.blocks();
  const auto initial_refinement_levels =
      domain_creator.initial_refinement_levels();
  const auto initial_extents = domain_creator.initial_extents();

  const size_t num_blocks = blocks.size();
  std::vector<std::vector<ElementId<Dim>>> element_ids_in_z_curve_order(
      num_blocks);
  for (size_t i = 0; i < num_blocks; i++) {
    element_ids_in_z_curve_order[i] =
        initial_element_ids(i, gsl::at(initial_refinement_levels, i), 0);
    alg::sort(element_ids_in_z_curve_order[i],
              [](const ElementId<Dim>& lhs, const ElementId<Dim>& rhs) {
                return domain::z_curve_index(lhs) < domain::z_curve_index(rhs);
              });
  }

  const auto costs = domain::get_element_costs(
      blocks, initial_refinement_levels, initial_extents, element_weight,
      Spectral::Quadrature::GaussLobatto);

  const domain::BlockZCurveProcDistribution<Dim> element_distribution(
      costs, number_of_procs_with_elements, blocks, initial_refinement_levels,
      initial_extents, global_procs_to_ignore);
  const auto proc_map = element_distribution.block_element_distribution();

  const size_t total_number_of_procs =
      number_of_procs_with_elements + global_procs_to_ignore.size();

  // whether or not we've assigned elements to a proc
  std::vector<bool> proc_hit(total_number_of_procs, false);

  size_t highest_proc_assigned = 0;
  for (size_t i = 0; i < num_blocks; i++) {
    size_t element_index = 0;
    const std::vector<std::pair<size_t, size_t>>& proc_map_this_block =
        proc_map[i];
    const size_t num_procs_this_block = proc_map_this_block.size();

    for (size_t j = 0; j < num_procs_this_block; j++) {
      const size_t expected_proc = proc_map_this_block[j].first;
      const size_t proc_allowance = proc_map_this_block[j].second;

      for (size_t k = 0; k < proc_allowance; k++) {
        const size_t actual_proc = element_distribution.get_proc_for_element(
            element_ids_in_z_curve_order[i][element_index]);
        // check that the correct processor is returned for the `ElementId`
        CHECK(actual_proc == expected_proc);
        proc_hit[actual_proc] = true;
        if (highest_proc_assigned < actual_proc) {
          highest_proc_assigned = actual_proc;
        }
      }
      element_index += proc_allowance;
    }
  }

  // check that ignored procs were indeed skipped and that all other procs
  // up to the highest one assigned were hit
  for (size_t i = 0; i < highest_proc_assigned + 1; i++) {
    if (global_procs_to_ignore.count(i) == 0) {
      CHECK(proc_hit[i]);
    } else {
      CHECK(not proc_hit[i]);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.ElementDistribution", "[Domain][Unit]") {
  // Test cost functions
  test_uniform_cost_function();
  test_weighted_cost_function(domain::ElementWeight::NumGridPoints);
  test_weighted_cost_function(
      domain::ElementWeight::NumGridPointsAndGridSpacing);

  // Inputs for testing `BlockZCurveProcDistribution`

  // 1D, single block
  const auto lattice_1d = domain::creators::AlignedLattice<1>(
      {{{{0.0, 1.0}}}}, {{4}}, {{6}}, {}, {}, {});
  // 2D
  const auto lattice_2d = domain::creators::AlignedLattice<2>(
      {{{{0.0, 0.3}}, {{0.0, 0.8, 2.5, 4.9}}}}, {{2, 3}}, {{4, 5}}, {}, {}, {});
  // 3D
  const auto lattice_3d = domain::creators::AlignedLattice<3>(
      {{{{0.0, 0.6}}, {{0.0, 0.4, 0.7}}, {{0.0, 0.3}}}}, {{2, 1, 3}},
      {{5, 4, 5}}, {}, {}, {});

  // Test element distribution construction logic

  // uniform distribution, single proc, single block
  test_uniform_element_distribution_construction(lattice_1d, 1);
  // uniform distribution, multiple procs, multiple blocks
  test_uniform_element_distribution_construction(
      lattice_2d, 20, std::unordered_set<size_t>{4, 20});
  // weighted distribution, multiple procs
  test_weighted_element_distribution_construction(
      domain::ElementWeight::NumGridPointsAndGridSpacing, lattice_3d, 22,
      std::unordered_set<size_t>{3, 4});
  // weighted distribution, more procs than elements to distribute
  test_weighted_element_distribution_construction(
      domain::ElementWeight::NumGridPoints, lattice_2d, 100,
      std::unordered_set<size_t>{0, 9});

  // Test processor retrieval with ignored processors
  test_proc_retrieval(domain::ElementWeight::NumGridPointsAndGridSpacing,
                      lattice_2d, 19, std::unordered_set<size_t>{0, 8, 9, 21});
  // Test processor retrieval when there are more processors requested than
  // `Element`s in the domain
  test_proc_retrieval(domain::ElementWeight::NumGridPointsAndGridSpacing,
                      lattice_2d, 100, std::unordered_set<size_t>{17});
}
