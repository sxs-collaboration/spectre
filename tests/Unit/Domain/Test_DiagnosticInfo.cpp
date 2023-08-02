// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/DiagnosticInfo.hpp"
#include "Domain/Domain.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"

namespace domain {
namespace {
struct EmptyMetavars {
  using component_list = tmpl::list<>;
};

void test_diagnostic_info() {
  std::vector<size_t> procs_per_node{2, 7, 5};
  Parallel::GlobalCache<EmptyMetavars> cache{
      tuples::TaggedTuple<>{}, {}, procs_per_node};

  const size_t number_of_cores = Parallel::number_of_procs<size_t>(cache);
  const size_t number_of_nodes = Parallel::number_of_nodes<size_t>(cache);

  std::vector<size_t> elements_per_core(number_of_cores, 0_st);
  std::vector<size_t> elements_per_node(number_of_nodes, 0_st);
  std::vector<size_t> grid_points_per_core(number_of_cores, 0_st);
  std::vector<size_t> grid_points_per_node(number_of_nodes, 0_st);

  alg::iota(elements_per_core, 1_st);
  alg::iota(grid_points_per_core, 10_st);

  size_t starting_core = 0;
  size_t ending_core = procs_per_node[0];
  for (size_t i = 0; i < procs_per_node.size(); i++) {
    ending_core = starting_core + procs_per_node[i];
    for (size_t j = starting_core; j < ending_core; j++) {
      elements_per_node[i] += elements_per_core[j];
      grid_points_per_node[i] += grid_points_per_core[j];
    }

    starting_core = ending_core;
  }

  const Domain<3> domain{std::vector<Block<3>>(13)};

  const std::string info =
      diagnostic_info(domain, cache, elements_per_core, elements_per_node,
                      grid_points_per_core, grid_points_per_node);

  const std::string expected_info =
      "----- Domain Info -----\n"
      "Total blocks: 13\n"
      "Total elements: 105\n"
      "Total grid points: 231\n"
      "Number of cores: 14\n"
      "Number of nodes: 3\n"
      "Elements per core: (1,2,3,4,5,6,7,8,9,10,11,12,13,14)\n"
      "Elements per node: (3,42,60)\n"
      "Grid points per core: (10,11,12,13,14,15,16,17,18,19,20,21,22,23)\n"
      "Grid points per node: (21,105,105)\n"
      "-----------------------\n";

  CHECK(info == expected_info);

#ifdef SPECTRE_DEBUG
  elements_per_core[0] += 1;
  CHECK_THROWS_WITH(
      ([&domain, &cache, &elements_per_core, &elements_per_node,
        &grid_points_per_core, &grid_points_per_node]() {
        [[maybe_unused]] const std::string bad_info =
            diagnostic_info(domain, cache, elements_per_core, elements_per_node,
                            grid_points_per_core, grid_points_per_node);
      })(),
      Catch::Matchers::ContainsSubstring(
          "Number of elements determined from elements_per_core ("));

  elements_per_core[0] -= 1;
  grid_points_per_core[0] += 1;
  CHECK_THROWS_WITH(
      ([&domain, &cache, &elements_per_core, &elements_per_node,
        &grid_points_per_core, &grid_points_per_node]() {
        [[maybe_unused]] const std::string bad_info =
            diagnostic_info(domain, cache, elements_per_core, elements_per_node,
                            grid_points_per_core, grid_points_per_node);
      })(),
      Catch::Matchers::ContainsSubstring(
          "Number of grid points determined from grid_points_per_core ("));

  grid_points_per_core[0] -= 1;
  // Pushing back 0 so the number of elements are the same, but the number of
  // cores is different, otherwise we would hit a different ASSERT first.
  elements_per_core.push_back(0);
  CHECK_THROWS_WITH(
      ([&domain, &cache, &elements_per_core, &elements_per_node,
        &grid_points_per_core, &grid_points_per_node]() {
        [[maybe_unused]] const std::string bad_info =
            diagnostic_info(domain, cache, elements_per_core, elements_per_node,
                            grid_points_per_core, grid_points_per_node);
      })(),
      Catch::Matchers::ContainsSubstring(
          "Number of cores determined from elements_per_core ("));

  elements_per_core.pop_back();
  grid_points_per_core.push_back(0);
  CHECK_THROWS_WITH(
      ([&domain, &cache, &elements_per_core, &elements_per_node,
        &grid_points_per_core, &grid_points_per_node]() {
        [[maybe_unused]] const std::string bad_info =
            diagnostic_info(domain, cache, elements_per_core, elements_per_node,
                            grid_points_per_core, grid_points_per_node);
      })(),
      Catch::Matchers::ContainsSubstring(
          "Number of cores determined from grid_points_per_core ("));

  grid_points_per_core.pop_back();
  elements_per_node.push_back(0);
  CHECK_THROWS_WITH(
      ([&domain, &cache, &elements_per_core, &elements_per_node,
        &grid_points_per_core, &grid_points_per_node]() {
        [[maybe_unused]] const std::string bad_info =
            diagnostic_info(domain, cache, elements_per_core, elements_per_node,
                            grid_points_per_core, grid_points_per_node);
      })(),
      Catch::Matchers::ContainsSubstring(
          "Number of nodes determined from elements_per_node ("));

  elements_per_node.pop_back();
  grid_points_per_node.push_back(0);
  CHECK_THROWS_WITH(
      ([&domain, &cache, &elements_per_core, &elements_per_node,
        &grid_points_per_core, &grid_points_per_node]() {
        [[maybe_unused]] const std::string bad_info =
            diagnostic_info(domain, cache, elements_per_core, elements_per_node,
                            grid_points_per_core, grid_points_per_node);
      })(),
      Catch::Matchers::ContainsSubstring(
          "Number of nodes determined from grid_points_per_node ("));
#endif
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.DiagnosticInfo", "[Unit][Domain]") {
  test_diagnostic_info();
}
}  // namespace domain
