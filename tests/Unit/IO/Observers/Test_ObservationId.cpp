// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Framework/TestHelpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"

namespace observers {
SPECTRE_TEST_CASE("Unit.IO.Observers.ObservationId", "[Unit][Observers]") {
  ObservationId id0(4., "ObservationType1");
  ObservationId id1(8., "ObservationType1");
  ObservationId id2(8., "ObservationType2");
  ObservationId id3(8., "subtype3");
  CHECK(id0 == id0);
  CHECK(id0.hash() == id0.hash());
  CHECK(id0.observation_key() == id0.observation_key());
  CHECK(id0 == ObservationId(4., "ObservationType1"));
  CHECK(id0 != id1);
  CHECK(id0.hash() != id1.hash());
  CHECK(id0.observation_key() == id1.observation_key());
  CHECK(id1 != id2);
  CHECK(id1.hash() != id2.hash());
  CHECK(id1.observation_key() != id2.observation_key());
  CHECK(id0.value() == 4.0);
  CHECK(id1.value() == 8.0);
  CHECK(id2.value() == 8.0);
  CHECK(id3 != id0);
  CHECK(id3 != id1);
  CHECK(id3 != id2);
  CHECK(id3.hash() != id2.hash());
  CHECK(id3.observation_key() != id2.observation_key());
  CHECK(id3.value() == 8.0);

  CHECK(get_output(id0) ==
        std::string(MakeString{} << '(' << id0.observation_key() << ","
                                 << id0.hash() << ',' << id0.value() << ')'));

  // Test PUP
  test_serialization(id0);
  test_serialization(id1);
  test_serialization(id2);
  test_serialization(id3);

  ObservationKey key0{"ObservationType1"};
  ObservationKey key1{"ObservationType2"};
  CHECK(key0 == key0);
  CHECK(key0 == ObservationKey{"ObservationType1"});
  CHECK(key0 != key1);
  CHECK(key0 == id0.observation_key());
  CHECK(get_output(key0) == "(" + get_output(key0.tag()) + ")");
  test_serialization(key0);
}
}  // namespace observers
