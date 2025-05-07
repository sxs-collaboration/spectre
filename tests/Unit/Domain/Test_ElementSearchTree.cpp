// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/ElementSearchTree.hpp"

namespace domain {

SPECTRE_TEST_CASE("Unit.Domain.ElementSearchTree", "[Domain][Unit]") {
  // [element_search_tree_example]
  const std::vector<ElementId<2>> element_ids{
      // Element layout in block-logical coordinates:
      //        xi -->
      //        -1     0       1
      // eta  -1  -------------
      //  |      |  0  |   2   |
      //  v    0 |-------------|
      //         |  1  | 3 | 4 |
      //       1  -------------
      ElementId<2>{0, {{{1, 0}, {1, 0}}}},  // 0
      ElementId<2>{0, {{{1, 0}, {1, 1}}}},  // 1
      ElementId<2>{0, {{{1, 1}, {1, 0}}}},  // 2
      ElementId<2>{0, {{{2, 2}, {1, 1}}}},  // 3
      ElementId<2>{0, {{{2, 3}, {1, 1}}}}   // 4
  };
  const ElementSearchTree<2> search_tree(element_ids);
  std::vector<ElementId<2>> search_result;
  search_tree.query(boost::geometry::index::covers(
                        tnsr::I<double, 2, Frame::BlockLogical>{{{0.5, -0.5}}}),
                    std::back_inserter(search_result));
  CHECK(search_result.size() == 1);
  CHECK(search_result[0] == element_ids[2]);
  // [element_search_tree_example]
  search_result.clear();
  search_tree.query(boost::geometry::index::covers(
                        tnsr::I<double, 2, Frame::BlockLogical>{{{0.0, -0.5}}}),
                    std::back_inserter(search_result));
  CHECK(search_result.size() == 2);
  CHECK(search_result[0] == element_ids[0]);
  CHECK(search_result[1] == element_ids[2]);

  {
    INFO("Multiple blocks");
    std::vector<ElementId<2>> more_element_ids(element_ids);
    more_element_ids.push_back(ElementId<2>{1, {{{1, 0}, {1, 0}}}});
    const auto search_trees = domain::index_element_ids<2>(more_element_ids);
    CHECK(search_trees.size() == 2);
    CHECK(search_trees.at(0).size() == 5);
    CHECK(search_trees.at(1).size() == 1);
    search_result.clear();
    search_trees.at(0).query(
        boost::geometry::index::covers(
            tnsr::I<double, 2, Frame::BlockLogical>{{{0.5, -0.5}}}),
        std::back_inserter(search_result));
    CHECK(search_result.size() == 1);
    CHECK(search_result[0] == element_ids[2]);
    search_result.clear();
    search_trees.at(1).query(
        boost::geometry::index::covers(
            tnsr::I<double, 2, Frame::BlockLogical>{{{-0.5, -0.5}}}),
        std::back_inserter(search_result));
    CHECK(search_result.size() == 1);
    CHECK(search_result[0] == more_element_ids.back());
    search_result.clear();
    search_trees.at(1).query(
        boost::geometry::index::covers(
            tnsr::I<double, 2, Frame::BlockLogical>{{{0.5, -0.5}}}),
        std::back_inserter(search_result));
    CHECK(search_result.empty());
  }
}

}  // namespace domain
