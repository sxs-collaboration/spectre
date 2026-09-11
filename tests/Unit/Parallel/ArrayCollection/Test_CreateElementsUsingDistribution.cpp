// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/TaggedTuple.hpp"
#include "Domain/Block.hpp"
#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Parallel/ArrayCollection/CreateElementsUsingDistribution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Metavars {
  using component_list = tmpl::list<>;
};

// Builds a domain with two blocks (stacked along dimension 1), each with the
// same refinement level and the same (DG) extents.
domain::creators::AlignedLattice<2> make_two_block_domain() {
  return domain::creators::AlignedLattice<2>(
      {{{{0.0, 1.0}}, {{0.0, 1.0, 2.0}}}}, {{{}, {}}}, {{{}, {}}}, {{2, 2}},
      {{3, 3}}, {}, {}, {});
}

// Records, for each `ElementId`, the target processor it was assigned to.
template <size_t Dim>
std::unordered_map<ElementId<Dim>, size_t> get_distribution(
    const domain::creators::AlignedLattice<Dim>& domain_creator,
    const std::optional<std::unordered_map<size_t, std::array<size_t, Dim>>>&
        weighting_extents_override) {
  const auto domain = domain_creator.create_domain();
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDelete)
  const Parallel::GlobalCache<Metavars> cache{tuples::TaggedTuple<>{},
                                              tuples::TaggedTuple<>{},
                                              std::vector<size_t>{2, 2}, 0, 0};

  std::unordered_map<ElementId<Dim>, size_t> element_to_proc{};
  Parallel::create_elements_using_distribution(
      [&element_to_proc](const ElementId<Dim>& element_id,
                         const size_t target_proc,
                         const size_t /*target_node*/) {
        element_to_proc[element_id] = target_proc;
      },
      std::optional{domain::ElementWeight::NumGridPoints}, domain.blocks(),
      domain_creator.initial_extents(),
      domain_creator.initial_refinement_levels(), Spectral::Basis::Legendre,
      Spectral::Quadrature::GaussLobatto, std::unordered_set<size_t>{},
      /*number_of_procs=*/4, /*number_of_nodes=*/2,
      /*num_of_procs_to_use=*/4, cache, false, weighting_extents_override);
  return element_to_proc;
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Parallel.ArrayCollection.CreateElementsUsingDistribution",
    "[Unit][Parallel]") {
  const auto domain_creator = make_two_block_domain();

  // Baseline: no override, both blocks weighted by their true (3x3) DG
  // extents.
  const auto baseline_distribution =
      get_distribution<2>(domain_creator, std::nullopt);

  // Override block 1's weighting extents to be much larger than its actual
  // (3x3) DG extents, e.g. as if it were being weighted by a much finer
  // subcell grid. This should change how elements are assigned to procs
  // relative to the baseline, since block 1's elements now dominate the
  // total cost.
  const std::unordered_map<size_t, std::array<size_t, 2>>
      weighting_extents_override{{1, {{9, 9}}}};
  const auto overridden_distribution =
      get_distribution<2>(domain_creator, weighting_extents_override);

  // Sanity check: both distributions cover the same set of elements.
  CHECK(baseline_distribution.size() == overridden_distribution.size());
  for (const auto& element_id_and_proc : baseline_distribution) {
    CHECK(overridden_distribution.count(element_id_and_proc.first) == 1);
  }

  // The two distributions should differ, since the override changes the
  // relative cost of block 1's elements compared to block 0's.
  CHECK(baseline_distribution != overridden_distribution);

  // Directly verify the override is equivalent to substituting the
  // overridden extents into `initial_extents` when computing the element
  // costs, which is what `create_elements_using_distribution` does
  // internally. Note that the production code passes the *true*
  // `initial_extents` to `BlockZCurveProcDistribution`, which only uses them
  // to sanity check their size, so that is what we do here too.
  const auto domain = domain_creator.create_domain();
  std::vector<std::array<size_t, 2>> manually_merged_extents =
      domain_creator.initial_extents();
  manually_merged_extents.at(1) = {{9, 9}};
  const auto expected_costs = domain::get_element_costs(
      domain.blocks(), domain_creator.initial_refinement_levels(),
      manually_merged_extents, domain::ElementWeight::NumGridPoints,
      Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const domain::BlockZCurveProcDistribution<2> expected_distribution{
      expected_costs,
      4,
      domain.blocks(),
      domain_creator.initial_refinement_levels(),
      domain_creator.initial_extents(),
      std::unordered_set<size_t>{}};
  for (const auto& [element_id, proc] : overridden_distribution) {
    CHECK(proc == expected_distribution.get_proc_for_element(element_id));
  }

  // `BlockZCurveProcDistribution` must not care which of the two extents it
  // is given, since the costs it distributes already account for the
  // override. The production code relies on this.
  const domain::BlockZCurveProcDistribution<2>
      expected_distribution_weighting_extents{
          expected_costs,          4,
          domain.blocks(),         domain_creator.initial_refinement_levels(),
          manually_merged_extents, std::unordered_set<size_t>{}};
  for (const auto& [element_id, proc] : overridden_distribution) {
    CHECK(proc == expected_distribution_weighting_extents.get_proc_for_element(
                      element_id));
  }
}
