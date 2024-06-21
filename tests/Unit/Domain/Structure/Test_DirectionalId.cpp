// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Framework/TestHelpers.hpp"

namespace {
void test_retrieval() {
  const auto check = []<size_t Dim>(const Direction<Dim>& direction,
                                    const ElementId<Dim>& element_id) {
    const auto check_impl = [&element_id,
                             &direction](const DirectionalId<Dim>& did) {
      CHECK(did.id() == element_id);
      CHECK(hash_value(did.id()) == hash_value(element_id));
      CHECK(hash_value(did) != hash_value(element_id));
      CHECK(did.direction() == direction);
      CHECK(hash_value(did.direction()) == hash_value(direction));
      CHECK(hash_value(did) != hash_value(direction));
    };
    const DirectionalId did0{direction, element_id};
    check_impl(did0);
    const DirectionalId did1 = serialize_and_deserialize(did0);
    check_impl(did1);
  };
  const std::array uls{Side::Upper, Side::Lower};
  std::vector element_ids_1d{
      ElementId<1>{0, {{{0, 0}}}}, ElementId<1>{0, {{{1, 0}}}},
      ElementId<1>{0, {{{1, 1}}}}, ElementId<1>{0, {{{2, 0}}}},
      ElementId<1>{0, {{{2, 1}}}}, ElementId<1>{0, {{{2, 3}}}}};
  for (const auto ul : uls) {
    INFO("Test 1d");
    for (size_t dim = 0; dim < 1; ++dim) {
      for (const auto& element_id : element_ids_1d) {
        check(Direction<1>(dim, ul), element_id);
      }
    }
  }

  const std::vector element_ids_2d{
      ElementId<2>{0, {{{0, 0}, {0, 0}}}}, ElementId<2>{0, {{{1, 0}, {0, 0}}}},
      ElementId<2>{0, {{{0, 0}, {1, 0}}}}, ElementId<2>{0, {{{1, 0}, {1, 0}}}},
      ElementId<2>{0, {{{1, 1}, {1, 0}}}}, ElementId<2>{0, {{{1, 0}, {1, 1}}}},
      ElementId<2>{0, {{{2, 0}, {1, 0}}}}, ElementId<2>{0, {{{2, 3}, {1, 0}}}},
      ElementId<2>{0, {{{1, 0}, {2, 0}}}}, ElementId<2>{0, {{{1, 0}, {2, 3}}}},
      ElementId<2>{0, {{{2, 1}, {1, 0}}}}, ElementId<2>{0, {{{1, 0}, {2, 2}}}},
      ElementId<2>{0, {{{2, 0}, {2, 0}}}}, ElementId<2>{0, {{{2, 0}, {2, 2}}}},
      ElementId<2>{0, {{{2, 2}, {2, 0}}}}, ElementId<2>{0, {{{2, 3}, {2, 0}}}},
      ElementId<2>{0, {{{2, 0}, {2, 3}}}}, ElementId<2>{0, {{{2, 3}, {2, 3}}}}};
  for (const auto ul : uls) {
    INFO("Test 2d");
    for (size_t dim = 0; dim < 2; ++dim) {
      for (const auto& element_id : element_ids_2d) {
        check(Direction<2>(dim, ul), element_id);
      }
    }
  }

  const std::vector element_ids_3d{ElementId<3>{0, {{{0, 0}, {0, 0}, {0, 0}}}},
                                   ElementId<3>{0, {{{1, 0}, {0, 0}, {0, 0}}}},
                                   ElementId<3>{0, {{{0, 0}, {1, 0}, {0, 0}}}},
                                   ElementId<3>{0, {{{0, 0}, {0, 0}, {1, 0}}}},
                                   ElementId<3>{0, {{{1, 1}, {0, 0}, {0, 0}}}},
                                   ElementId<3>{0, {{{0, 0}, {1, 1}, {0, 0}}}},
                                   ElementId<3>{0, {{{0, 0}, {0, 0}, {1, 1}}}},
                                   ElementId<3>{0, {{{1, 0}, {1, 0}, {0, 0}}}},
                                   ElementId<3>{0, {{{1, 0}, {0, 0}, {1, 0}}}},
                                   ElementId<3>{0, {{{1, 0}, {1, 0}, {1, 0}}}},
                                   ElementId<3>{0, {{{1, 1}, {1, 0}, {1, 0}}}},
                                   ElementId<3>{0, {{{1, 0}, {1, 1}, {1, 0}}}},
                                   ElementId<3>{0, {{{1, 0}, {1, 0}, {1, 1}}}},
                                   ElementId<3>{0, {{{2, 0}, {1, 0}, {1, 0}}}},
                                   ElementId<3>{0, {{{1, 0}, {2, 0}, {1, 0}}}},
                                   ElementId<3>{0, {{{1, 0}, {1, 0}, {2, 0}}}},
                                   ElementId<3>{0, {{{2, 3}, {1, 0}, {1, 0}}}},
                                   ElementId<3>{0, {{{1, 0}, {2, 3}, {1, 0}}}},
                                   ElementId<3>{0, {{{1, 0}, {1, 0}, {2, 3}}}},
                                   ElementId<3>{0, {{{2, 1}, {1, 0}, {1, 0}}}},
                                   ElementId<3>{0, {{{1, 0}, {2, 1}, {1, 0}}}},
                                   ElementId<3>{0, {{{1, 0}, {1, 0}, {2, 1}}}},
                                   ElementId<3>{0, {{{2, 1}, {2, 0}, {1, 0}}}},
                                   ElementId<3>{0, {{{2, 1}, {1, 0}, {2, 0}}}},
                                   ElementId<3>{0, {{{2, 0}, {2, 1}, {1, 0}}}},
                                   ElementId<3>{0, {{{1, 0}, {2, 1}, {2, 0}}}},
                                   ElementId<3>{0, {{{2, 0}, {1, 0}, {2, 1}}}},
                                   ElementId<3>{0, {{{1, 0}, {2, 0}, {2, 1}}}},
                                   ElementId<3>{0, {{{2, 0}, {2, 0}, {2, 1}}}},
                                   ElementId<3>{0, {{{2, 1}, {2, 0}, {2, 1}}}},
                                   ElementId<3>{0, {{{2, 0}, {2, 1}, {2, 1}}}},
                                   ElementId<3>{0, {{{2, 1}, {2, 1}, {2, 0}}}},
                                   ElementId<3>{0, {{{2, 1}, {2, 1}, {2, 1}}}}};
  for (const auto ul : uls) {
    INFO("Test 3d");
    for (size_t dim = 0; dim < 3; ++dim) {
      for (const auto& element_id : element_ids_3d) {
        check(Direction<3>(dim, ul), element_id);
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Structure.DirectionalId", "[Domain][Unit]") {
  test_retrieval();
}
