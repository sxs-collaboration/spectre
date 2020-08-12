// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "Domain/Structure/Hypercube.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"

template <size_t HypercubeDim, size_t ElementDim, size_t ExpectedSize>
void test_hypercube_iterator(
    std::array<HypercubeElement<ElementDim, HypercubeDim>, ExpectedSize>
        expected_elements) {
  CAPTURE(ElementDim);
  CAPTURE(HypercubeDim);
  HypercubeElementsIterator<ElementDim, HypercubeDim> elements_iterator{};
  static_assert(ExpectedSize > 0);
  static_assert(elements_iterator.size() == ExpectedSize);
  CHECK(elements_iterator ==
        HypercubeElementsIterator<ElementDim, HypercubeDim>::begin());
  CHECK(*elements_iterator ==
        *HypercubeElementsIterator<ElementDim, HypercubeDim>::begin());
  CHECK(elements_iterator !=
        HypercubeElementsIterator<ElementDim, HypercubeDim>::end());
  size_t i = 0;
  for (const auto& element : elements_iterator) {
    CAPTURE(i);
    CAPTURE(element);
    CHECK(element == gsl::at(expected_elements, i));
    ++i;
  }
  CHECK(i == ExpectedSize);
  // Test postfix operator
  elements_iterator =
      HypercubeElementsIterator<ElementDim, HypercubeDim>::begin();
  const auto previous_iterator = elements_iterator++;
  CHECK(previous_iterator ==
        HypercubeElementsIterator<ElementDim, HypercubeDim>::begin());
  CHECK(elements_iterator ==
        ++HypercubeElementsIterator<ElementDim, HypercubeDim>::begin());
}

SPECTRE_TEST_CASE("Unit.Domain.Hypercube", "[Domain][Unit]") {
  {
    const Vertex<0> point{};
    CHECK(get_output(point) == "Vertex0D[(),()]");
    CHECK(point.dimensions_in_parent() == std::array<size_t, 0>{});
    CHECK(point.index() == std::array<Side, 0>{});
    CHECK(point == Vertex<0>{});
    const Vertex<1> left{Side::Lower};
    CHECK(get_output(left) == "Vertex1D[(),(Lower)]");
    CHECK(left.index() == std::array<Side, 1>{{Side::Lower}});
    CHECK(left.side() == Side::Lower);
    CHECK(left == Vertex<1>{Side::Lower});
    CHECK(left != Vertex<1>{Side::Upper});
    const Vertex<2> lower_right{Side::Lower, Side::Upper};
    CHECK(get_output(lower_right) == "Vertex2D[(),(Lower,Upper)]");
    CHECK(lower_right.index() ==
          std::array<Side, 2>{{Side::Lower, Side::Upper}});
    CHECK(lower_right == Vertex<2>{{{Side::Lower, Side::Upper}}});
    CHECK(lower_right != Vertex<2>{{{Side::Upper, Side::Upper}}});
  }
  {
    const Edge<2> south{0, {{Side::Lower}}};
    CHECK(get_output(south) == "Edge2D[(0),(Lower)]");
    CHECK(south.dimensions_in_parent() == std::array<size_t, 1>{{0}});
    CHECK(south.dimension_in_parent() == 0);
    CHECK(south.index() == std::array<Side, 1>{{Side::Lower}});
    CHECK(south.side_in_parent_dimension(1) == Side::Lower);
    CHECK(south.side() == Side::Lower);
    CHECK(south == Edge<2>{0, {{Side::Lower}}});
    CHECK(south != Edge<2>{1, {{Side::Lower}}});
    CHECK(south != Edge<2>{0, {{Side::Upper}}});
    const Edge<2> north{0, {{Side::Upper}}};
    CHECK(get_output(north) == "Edge2D[(0),(Upper)]");
    CHECK(north.dimensions_in_parent() == std::array<size_t, 1>{{0}});
    CHECK(north.dimension_in_parent() == 0);
    CHECK(north.index() == std::array<Side, 1>{{Side::Upper}});
    CHECK(north.side_in_parent_dimension(1) == Side::Upper);
    CHECK(north.side() == Side::Upper);
    const Edge<2> west{1, {{Side::Lower}}};
    CHECK(get_output(west) == "Edge2D[(1),(Lower)]");
    CHECK(west.dimensions_in_parent() == std::array<size_t, 1>{{1}});
    CHECK(west.dimension_in_parent() == 1);
    CHECK(west.index() == std::array<Side, 1>{{Side::Lower}});
    CHECK(west.side_in_parent_dimension(0) == Side::Lower);
    const Edge<2> east{1, {{Side::Upper}}};
    CHECK(get_output(east) == "Edge2D[(1),(Upper)]");
    CHECK(east.dimensions_in_parent() == std::array<size_t, 1>{{1}});
    CHECK(east.dimension_in_parent() == 1);
    CHECK(east.index() == std::array<Side, 1>{{Side::Upper}});
    CHECK(east.side_in_parent_dimension(0) == Side::Upper);
    CHECK(east.side() == Side::Upper);
  }
  {
    const Edge<3> top_left{1, {{Side::Lower, Side::Upper}}};
    CHECK(get_output(top_left) == "Edge3D[(1),(Lower,Upper)]");
    CHECK(top_left.dimensions_in_parent() == std::array<size_t, 1>{{1}});
    CHECK(top_left.dimension_in_parent() == 1);
    CHECK(top_left.index() == std::array<Side, 2>{{Side::Lower, Side::Upper}});
    CHECK(top_left.side_in_parent_dimension(0) == Side::Lower);
    CHECK(top_left.side_in_parent_dimension(2) == Side::Upper);
    CHECK(top_left == Edge<3>{1, {{Side::Lower, Side::Upper}}});
    CHECK(top_left != Edge<3>{0, {{Side::Lower, Side::Upper}}});
    CHECK(top_left != Edge<3>{1, {{Side::Lower, Side::Lower}}});
    const Edge<3> top_front{0, {{Side::Lower, Side::Upper}}};
    CHECK(get_output(top_front) == "Edge3D[(0),(Lower,Upper)]");
    CHECK(top_front.dimensions_in_parent() == std::array<size_t, 1>{{0}});
    CHECK(top_front.dimension_in_parent() == 0);
    CHECK(top_front.index() == std::array<Side, 2>{{Side::Lower, Side::Upper}});
    CHECK(top_front.side_in_parent_dimension(1) == Side::Lower);
    CHECK(top_front.side_in_parent_dimension(2) == Side::Upper);
    const Edge<3> front_left{2, {{Side::Lower, Side::Upper}}};
    CHECK(get_output(front_left) == "Edge3D[(2),(Lower,Upper)]");
    CHECK(front_left.dimensions_in_parent() == std::array<size_t, 1>{{2}});
    CHECK(front_left.dimension_in_parent() == 2);
    CHECK(front_left.index() ==
          std::array<Side, 2>{{Side::Lower, Side::Upper}});
    CHECK(front_left.side_in_parent_dimension(0) == Side::Lower);
    CHECK(front_left.side_in_parent_dimension(1) == Side::Upper);
  }
  {
    const Face<3> top{{{0, 1}}, {{Side::Upper}}};
    CHECK(get_output(top) == "Face3D[(0,1),(Upper)]");
    CHECK(top.dimensions_in_parent() == std::array<size_t, 2>{{0, 1}});
    CHECK(top.index() == std::array<Side, 1>{{Side::Upper}});
    CHECK(top.side_in_parent_dimension(2) == Side::Upper);
    CHECK(top.side() == Side::Upper);
    CHECK(top == Face<3>{{{0, 1}}, {{Side::Upper}}});
    CHECK(top != Face<3>{{{0, 2}}, {{Side::Upper}}});
    CHECK(top != Face<3>{{{0, 1}}, {{Side::Lower}}});
    const Face<3> top_rotated{{{1, 0}}, {{Side::Upper}}};
    CHECK(top_rotated == top);
    const Face<3> left{{{1, 2}}, {{Side::Lower}}};
    CHECK(get_output(left) == "Face3D[(1,2),(Lower)]");
    CHECK(left.dimensions_in_parent() == std::array<size_t, 2>{{1, 2}});
    CHECK(left.index() == std::array<Side, 1>{{Side::Lower}});
    CHECK(left.side_in_parent_dimension(0) == Side::Lower);
    CHECK(left.side() == Side::Lower);
    const Face<3> left_rotated{{{2, 1}}, {{Side::Lower}}};
    CHECK(left_rotated == left);
    const Face<3> front{{{0, 2}}, {{Side::Lower}}};
    CHECK(get_output(front) == "Face3D[(0,2),(Lower)]");
    CHECK(front.dimensions_in_parent() == std::array<size_t, 2>{{0, 2}});
    CHECK(front.index() == std::array<Side, 1>{{Side::Lower}});
    CHECK(front.side_in_parent_dimension(1) == Side::Lower);
    CHECK(front.side() == Side::Lower);
    const Face<3> front_rotated{{{2, 0}}, {{Side::Lower}}};
    CHECK(front_rotated == front);
  }
  {
    const Cell<3> cube{};
    CHECK(get_output(cube) == "Cell3D[(0,1,2),()]");
    CHECK(cube.dimensions_in_parent() == std::array<size_t, 3>{{0, 1, 2}});
    CHECK(cube.index() == std::array<Side, 0>{});
    const Cell<3> cube_rotated{{{1, 2, 0}}, {}};
    CHECK(cube_rotated == cube);
  }
  {
    INFO("Hypercube iterator");
    // 0D: .
    // -> 1 vertex
    test_hypercube_iterator<0, 0, 1>({{Vertex<0>{}}});
    // 1D: -
    // -> 2 vertices and 1 edge
    test_hypercube_iterator<1, 0, 2>(
        {{Vertex<1>{Side::Lower}, Vertex<1>{Side::Upper}}});
    test_hypercube_iterator<1, 1, 1>({{Edge<1>{}}});
    // 2D:
    // +---+
    // |   |
    // +---+
    // -> 4 vertices, 4 edges and 1 face
    test_hypercube_iterator<2, 0, 4>({{Vertex<2>{Side::Lower, Side::Lower},
                                       Vertex<2>{Side::Upper, Side::Lower},
                                       Vertex<2>{Side::Lower, Side::Upper},
                                       Vertex<2>{Side::Upper, Side::Upper}}});
    test_hypercube_iterator<2, 1, 4>(
        {{Edge<2>{0, {{Side::Lower}}}, Edge<2>{0, {{Side::Upper}}},
          Edge<2>{1, {{Side::Lower}}}, Edge<2>{1, {{Side::Upper}}}}});
    test_hypercube_iterator<2, 2, 1>({{Face<2>{}}});
    // 3D:
    //    +---+
    //  /   / |
    // +---+  +
    // |   | /
    // +---+
    // -> 8 vertices, 12 edges, 6 faces and 1 cell
    test_hypercube_iterator<3, 0, 8>(
        {{Vertex<3>{Side::Lower, Side::Lower, Side::Lower},
          Vertex<3>{Side::Upper, Side::Lower, Side::Lower},
          Vertex<3>{Side::Lower, Side::Upper, Side::Lower},
          Vertex<3>{Side::Upper, Side::Upper, Side::Lower},
          Vertex<3>{Side::Lower, Side::Lower, Side::Upper},
          Vertex<3>{Side::Upper, Side::Lower, Side::Upper},
          Vertex<3>{Side::Lower, Side::Upper, Side::Upper},
          Vertex<3>{Side::Upper, Side::Upper, Side::Upper}}});
    test_hypercube_iterator<3, 1, 12>(
        {{Edge<3>{0, {{Side::Lower, Side::Lower}}},
          Edge<3>{0, {{Side::Upper, Side::Lower}}},
          Edge<3>{0, {{Side::Lower, Side::Upper}}},
          Edge<3>{0, {{Side::Upper, Side::Upper}}},
          Edge<3>{1, {{Side::Lower, Side::Lower}}},
          Edge<3>{1, {{Side::Upper, Side::Lower}}},
          Edge<3>{1, {{Side::Lower, Side::Upper}}},
          Edge<3>{1, {{Side::Upper, Side::Upper}}},
          Edge<3>{2, {{Side::Lower, Side::Lower}}},
          Edge<3>{2, {{Side::Upper, Side::Lower}}},
          Edge<3>{2, {{Side::Lower, Side::Upper}}},
          Edge<3>{2, {{Side::Upper, Side::Upper}}}}});
    test_hypercube_iterator<3, 2, 6>({{Face<3>{{{0, 1}}, {{Side::Lower}}},
                                       Face<3>{{{0, 1}}, {{Side::Upper}}},
                                       Face<3>{{{0, 2}}, {{Side::Lower}}},
                                       Face<3>{{{0, 2}}, {{Side::Upper}}},
                                       Face<3>{{{1, 2}}, {{Side::Lower}}},
                                       Face<3>{{{1, 2}}, {{Side::Upper}}}}});
    test_hypercube_iterator<3, 3, 1>({{Cell<3>{}}});
  }
}
