// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <functional>
#include <string>

#include "DataStructures/PrimitiveId.hpp"
#include "Utilities/GetOutput.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.PrimitiveId", "[Unit][DataStructures]") {
  {
    const PrimitiveId<size_t> id{2};
    CHECK(id == 2);
    using Hash = std::hash<PrimitiveId<size_t>>;
    CHECK(Hash{}(id) == Hash{}(PrimitiveId<size_t>{2}));
    CHECK(get_output(id) == "2");
    check_cmp(PrimitiveId<size_t>{1}, PrimitiveId<size_t>{2});
  }
  {
    const PrimitiveId<double> id{2.};
    CHECK(id == 2.);
    using Hash = std::hash<PrimitiveId<double>>;
    CHECK(Hash{}(id) == Hash{}(PrimitiveId<double>{2.}));
    CHECK(get_output(id) == "2.0");
    check_cmp(PrimitiveId<double>{1.}, PrimitiveId<double>{2.});
  }
}
