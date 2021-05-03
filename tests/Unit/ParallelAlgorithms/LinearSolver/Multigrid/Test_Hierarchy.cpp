// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <ostream>
#include <unordered_set>
#include <vector>

#include "Domain/Structure/ElementId.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Hierarchy.hpp"
#include "Utilities/StdHelpers.hpp"

namespace LinearSolver::multigrid {

SPECTRE_TEST_CASE("Unit.ParallelMultigrid.Hierarchy",
                  "[Unit][ParallelAlgorithms][LinearSolver]") {
  {
    INFO("Coarsening");
    CHECK(coarsen<1>({{{0}}, {{1}}, {{2}}}) ==
          std::vector<std::array<size_t, 1>>{{{0}}, {{0}}, {{1}}});
    CHECK(coarsen<2>({{{0, 1}}, {{1, 1}}, {{2, 0}}, {{3, 4}}}) ==
          std::vector<std::array<size_t, 2>>{
              {{0, 0}}, {{0, 0}}, {{1, 0}}, {{2, 3}}});
    CHECK(coarsen<3>({{{0, 1, 2}}, {{1, 1, 1}}, {{2, 0, 1}}, {{3, 4, 5}}}) ==
          std::vector<std::array<size_t, 3>>{
              {{0, 0, 1}}, {{0, 0, 0}}, {{1, 0, 0}}, {{2, 3, 4}}});
  }
  {
    INFO("Parent ID");
    const size_t block_id = 3;
    CHECK(parent_id<1>({block_id, {{{0, 0}}}, 0}) ==
          ElementId<1>{block_id, {{{0, 0}}}, 1});
    CHECK(parent_id<1>({block_id, {{{1, 0}}}, 0}) ==
          ElementId<1>{block_id, {{{0, 0}}}, 1});
    CHECK(parent_id<1>({block_id, {{{1, 1}}}, 0}) ==
          ElementId<1>{block_id, {{{0, 0}}}, 1});
    CHECK(parent_id<1>({block_id, {{{2, 2}}}, 2}) ==
          ElementId<1>{block_id, {{{1, 1}}}, 3});
    CHECK(parent_id<1>({block_id, {{{2, 3}}}, 2}) ==
          ElementId<1>{block_id, {{{1, 1}}}, 3});
    CHECK(parent_id<2>({block_id, {{{0, 0}, {1, 0}}}, 0}) ==
          ElementId<2>{block_id, {{{0, 0}, {0, 0}}}, 1});
    CHECK(parent_id<2>({block_id, {{{2, 2}, {1, 1}}}, 2}) ==
          ElementId<2>{block_id, {{{1, 1}, {0, 0}}}, 3});
    CHECK(parent_id<3>({block_id, {{{0, 0}, {1, 0}, {2, 1}}}, 2}) ==
          ElementId<3>{block_id, {{{0, 0}, {0, 0}, {1, 0}}}, 3});
  }
  {
    INFO("Child IDs");
    const size_t block_id = 3;
    CHECK(child_ids<1>({block_id, {{{0, 0}}}, 0}, {{0}}).empty());
    CHECK(child_ids<1>({block_id, {{{0, 0}}}, 1}, {{0}}) ==
          std::unordered_set<ElementId<1>>{{block_id, {{{0, 0}}}, 0}});
    CHECK(child_ids<1>({block_id, {{{0, 0}}}, 1}, {{1}}) ==
          std::unordered_set<ElementId<1>>{{block_id, {{{1, 0}}}, 0},
                                           {block_id, {{{1, 1}}}, 0}});
    CHECK(child_ids<1>({block_id, {{{2, 2}}}, 2}, {{3}}) ==
          std::unordered_set<ElementId<1>>{{block_id, {{{3, 4}}}, 1},
                                           {block_id, {{{3, 5}}}, 1}});
    CHECK(child_ids<2>({block_id, {{{0, 0}, {1, 0}}}, 0}, {{0, 1}}).empty());
    CHECK(child_ids<2>({block_id, {{{0, 0}, {1, 0}}}, 1}, {{0, 1}}) ==
          std::unordered_set<ElementId<2>>{{block_id, {{{0, 0}, {1, 0}}}, 0}});
    CHECK(child_ids<2>({block_id, {{{0, 0}, {1, 0}}}, 2}, {{1, 1}}) ==
          std::unordered_set<ElementId<2>>{{block_id, {{{1, 0}, {1, 0}}}, 1},
                                           {block_id, {{{1, 1}, {1, 0}}}, 1}});
    CHECK(child_ids<2>({block_id, {{{0, 0}, {1, 0}}}, 2}, {{1, 1}}) ==
          std::unordered_set<ElementId<2>>{{block_id, {{{1, 0}, {1, 0}}}, 1},
                                           {block_id, {{{1, 1}, {1, 0}}}, 1}});
    CHECK(child_ids<2>({block_id, {{{0, 0}, {1, 0}}}, 2}, {{1, 2}}) ==
          std::unordered_set<ElementId<2>>{{block_id, {{{1, 0}, {2, 0}}}, 1},
                                           {block_id, {{{1, 1}, {2, 0}}}, 1},
                                           {block_id, {{{1, 0}, {2, 1}}}, 1},
                                           {block_id, {{{1, 1}, {2, 1}}}, 1}});
    CHECK(child_ids<3>({block_id, {{{0, 0}, {1, 0}, {2, 0}}}, 0}, {{0, 1, 2}})
              .empty());
    CHECK(
        child_ids<3>({block_id, {{{0, 0}, {1, 0}, {2, 0}}}, 1}, {{0, 1, 2}}) ==
        std::unordered_set<ElementId<3>>{
            {block_id, {{{0, 0}, {1, 0}, {2, 0}}}, 0}});
    CHECK(
        child_ids<3>({block_id, {{{0, 0}, {1, 1}, {2, 0}}}, 2}, {{1, 2, 2}}) ==
        std::unordered_set<ElementId<3>>{
            {block_id, {{{1, 0}, {2, 2}, {2, 0}}}, 1},
            {block_id, {{{1, 1}, {2, 2}, {2, 0}}}, 1},
            {block_id, {{{1, 0}, {2, 3}, {2, 0}}}, 1},
            {block_id, {{{1, 1}, {2, 3}, {2, 0}}}, 1}});
    CHECK(
        child_ids<3>({block_id, {{{0, 0}, {1, 1}, {2, 0}}}, 2}, {{1, 2, 3}}) ==
        std::unordered_set<ElementId<3>>{
            {block_id, {{{1, 0}, {2, 2}, {3, 0}}}, 1},
            {block_id, {{{1, 1}, {2, 2}, {3, 0}}}, 1},
            {block_id, {{{1, 0}, {2, 3}, {3, 0}}}, 1},
            {block_id, {{{1, 1}, {2, 3}, {3, 0}}}, 1},
            {block_id, {{{1, 0}, {2, 2}, {3, 1}}}, 1},
            {block_id, {{{1, 1}, {2, 2}, {3, 1}}}, 1},
            {block_id, {{{1, 0}, {2, 3}, {3, 1}}}, 1},
            {block_id, {{{1, 1}, {2, 3}, {3, 1}}}, 1}});
  }
}

}  // namespace LinearSolver::multigrid
