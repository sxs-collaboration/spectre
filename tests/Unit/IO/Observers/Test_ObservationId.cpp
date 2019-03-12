// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "IO/Observer/ObservationId.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace observer_testing_detail {
struct ObservationType1 {};
struct ObservationType2 {};
}  // namespace observer_testing_detail

namespace observers {
SPECTRE_TEST_CASE("Unit.IO.Observers.ObservationId", "[Unit][Observers]") {
  ObservationId id0(4., observer_testing_detail::ObservationType1{});
  ObservationId id1(8., observer_testing_detail::ObservationType1{});
  ObservationId id2(8., observer_testing_detail::ObservationType2{});
  ObservationId id3(8., observer_testing_detail::ObservationType2{},
                    "subtype1");
  ObservationId id4(8., observer_testing_detail::ObservationType2{},
                    "subtype2");
  CHECK(id0 == id0);
  CHECK(id0.hash() == id0.hash());
  CHECK(id0.observation_type_hash() == id0.observation_type_hash());
  CHECK(id0 == ObservationId(4., observer_testing_detail::ObservationType1{}));
  CHECK(id0 != id1);
  CHECK(id0.hash() != id1.hash());
  CHECK(id0.observation_type_hash() == id1.observation_type_hash());
  CHECK(id1 != id2);
  CHECK(id1.hash() != id2.hash());
  CHECK(id1.observation_type_hash() != id2.observation_type_hash());
  CHECK(id0.value() == 4.0);
  CHECK(id1.value() == 8.0);
  CHECK(id2.value() == 8.0);
  CHECK(id3 != id0);
  CHECK(id3 != id1);
  CHECK(id3 != id2);
  CHECK(id3.hash() != id2.hash());
  CHECK(id3.observation_type_hash() != id2.observation_type_hash());
  CHECK(id3.value() == 8.0);
  CHECK(id4 != id0);
  CHECK(id4 != id1);
  CHECK(id4 != id2);
  CHECK(id4 != id3);
  CHECK(id4.hash() != id3.hash());
  CHECK(id4.observation_type_hash() != id3.observation_type_hash());
  CHECK(id4.value() == 8.0);

  CHECK(get_output(id0) ==
        std::string(MakeString{} << '(' << id0.observation_type_hash() << ","
                                 << id0.hash() << ',' << id0.value() << ')'));

  // Test PUP
  test_serialization(id0);
  test_serialization(id1);
  test_serialization(id2);
  test_serialization(id3);
  test_serialization(id4);
}
}  // namespace observers
