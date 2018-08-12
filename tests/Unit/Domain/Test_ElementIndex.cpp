// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstring>
#include <functional>
#include <new>
#include <string>

#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/SegmentId.hpp"  // IWYU pragma: keep
#include "Utilities/GetOutput.hpp"

namespace {
template <size_t VolumeDim>
void check(const size_t block1,
           const std::array<SegmentId, VolumeDim>& segments1,
           const size_t block2,
           const std::array<SegmentId, VolumeDim>& segments2) {
  using Hash = std::hash<ElementIndex<VolumeDim>>;

  const ElementId<VolumeDim> id1(block1, segments1);
  const ElementId<VolumeDim> id2(block2, segments2);

  ElementIndex<VolumeDim> element_index1{}, element_index2{};
  // Check for nondeterminacy due to previous memory state.
  std::memset(&element_index1, 0, sizeof(element_index1));
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ > 7)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif  // defined(__GNUC__) && !defined(__clang__) && (__GNUC__ > 7)
  std::memset(&element_index2, 255, sizeof(element_index2));
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ > 7)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__) && (__GNUC__ > 7)
  new (&element_index1) ElementIndex<VolumeDim>(id1);
  new (&element_index2) ElementIndex<VolumeDim>(id2);

  CHECK((element_index1 == element_index2) == (id1 == id2));
  CHECK((element_index1 != element_index2) == (id1 != id2));
  CHECK((Hash{}(element_index1) == Hash{}(element_index2)) == (id1 == id2));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.ElementIndex", "[Domain][Unit]") {
  const std::array<size_t, 3> blocks{{0, 1, 4}};
  const std::array<SegmentId, 4> segments{{{0, 0}, {1, 0}, {1, 1}, {8, 4}}};

  for (const auto& block1 : blocks) {
    for (const auto& block2 : blocks) {
      for (const auto& segment10 : segments) {
        for (const auto& segment20 : segments) {
          check<1>(block1, {{segment10}}, block2, {{segment20}});
          for (const auto& segment11 : segments) {
            for (const auto& segment21 : segments) {
              check<2>(block1, {{segment10, segment11}},
                       block2, {{segment20, segment21}});
              for (const auto& segment12 : segments) {
                for (const auto& segment22 : segments) {
                  check<3>(block1, {{segment10, segment11, segment12}},
                           block2, {{segment20, segment21, segment22}});
                }
              }
            }
          }
        }
      }
    }
  }

  CHECK(get_output(ElementIndex<1>(ElementId<1>(3, {{{4, 5}}})))
        == "[3:4:5]");
  CHECK(get_output(ElementIndex<2>(ElementId<2>(3, {{{4, 5}, {6, 7}}})))
        == "[3:4:5][3:6:7]");
  CHECK(get_output(ElementIndex<3>(ElementId<3>(3, {{{4, 5}, {6, 7}, {8, 9}}})))
        == "[3:4:5][3:6:7][3:8:9]");
}
