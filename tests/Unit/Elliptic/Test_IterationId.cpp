// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <functional>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Elliptic/IterationId.hpp"
#include "Elliptic/Tags.hpp"
#include "Utilities/GetOutput.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
struct MajorId {
  using type = size_t;
};
struct MinorId {
  using type = size_t;
}
}  // namespace

namespace Elliptic {
template <>
constexpr size_t iteration_id_value_factor<MajorId> = 10;
template <>
constexpr size_t iteration_id_value_factor<MinorId> = 1;
}  // namespace Elliptic

SPECTRE_TEST_CASE("Unit.Elliptic.IterationId", "[Unit][Elliptic]") {
  using iteration_id = Elliptic::IterationId<MajorId, MinorId>;
  {
    INFO("Comparison logic");
    check_cmp(iteration_id{1_st, 1_st}, iteration_id{1_st, 2_st});
    check_cmp(iteration_id{1_st, 1_st}, iteration_id{2_st, 1_st});
    check_cmp(iteration_id{1_st, 1_st}, iteration_id{2_st, 0_st});
  }
  {
    INFO("Hash");
    using Hash = std::hash<iteration_id>;
    CHECK(Hash{}(iteration_id{1_st, 2_st}) == Hash{}(iteration_id{1_st, 2_st}));
    CHECK(Hash{}(iteration_id{1_st, 2_st}) != Hash{}(iteration_id{2_st, 2_st}));
    CHECK(Hash{}(iteration_id{1_st, 2_st}) != Hash{}(iteration_id{1_st, 1_st}));
  }
  {
    INFO("Value");
    CHECK(iteration_id{3_st, 2_st}.value() == 32.);
  }
  {
    INFO("Increment");
    CHECK(iteration_id{1_st, 2_st}.increment<MinorId>() ==
          iteration_id{1_st, 3_st});
    CHECK(iteration_id{1_st, 2_st}.increment<MajorId>() ==
          iteration_id{2_st, 0_st});
  }
}
