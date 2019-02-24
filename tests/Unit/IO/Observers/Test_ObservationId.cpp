// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <functional>
#include <string>

#include "IO/Observer/ObservationId.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace observer_testing_detail {
class DummyTimeId {
 public:
  explicit DummyTimeId(const size_t value) : value_(value) {}
  size_t value() const { return value_; }

 private:
  size_t value_;
};
size_t hash_value(const DummyTimeId& id) noexcept { return id.value(); }

struct ObservationType1 {};
struct ObservationType2 {};

}  // namespace observer_testing_detail

namespace std {
template <>
struct hash<observer_testing_detail::DummyTimeId> {
  size_t operator()(const observer_testing_detail::DummyTimeId& id) const
      noexcept {
    return id.value();
  }
};
}  // namespace std

using DummyTimeId = observer_testing_detail::DummyTimeId;
namespace observers {
SPECTRE_TEST_CASE("Unit.IO.Observers.ObservationId", "[Unit][Observers]") {
  ObservationId id0(DummyTimeId(4),
                    observer_testing_detail::ObservationType1{});
  ObservationId id1(DummyTimeId(8),
                    observer_testing_detail::ObservationType1{});
  ObservationId id2(DummyTimeId(8),
                    observer_testing_detail::ObservationType2{});
  ObservationId id3(DummyTimeId(8), observer_testing_detail::ObservationType2{},
                    "subtype1");
  ObservationId id4(DummyTimeId(8), observer_testing_detail::ObservationType2{},
                    "subtype2");
  CHECK(id0 == id0);
  CHECK(id0.hash() == id0.hash());
  CHECK(id0.observation_type_hash() == id0.observation_type_hash());
  CHECK(id0 == ObservationId(DummyTimeId(4),
                             observer_testing_detail::ObservationType1{}));
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

  ObservationId time0(Time(Slab{0.0, 1.0}, 0),
                      observer_testing_detail::ObservationType1{});
  ObservationId time1(Time(Slab{0.0, 1.0}, Time::rational_t(1, 2)),
                      observer_testing_detail::ObservationType1{});
  ObservationId time2(Time(Slab{0.0, 1.0}, 0),
                      observer_testing_detail::ObservationType2{});
  CHECK(time0 != id0);
  CHECK(time0.hash() != id0.hash());
  CHECK(time0 == time0);
  CHECK(time0.hash() == time0.hash());
  CHECK(time0.observation_type_hash() == time0.observation_type_hash());
  CHECK(time0 != time1);
  CHECK(time0.hash() != time1.hash());
  CHECK(time0.observation_type_hash() == time1.observation_type_hash());
  CHECK(time0.value() == 0.0);
  CHECK(time1.value() == 0.5);
  CHECK(time2.value() == 0.0);
  CHECK(time0 != time2);
  CHECK(time0.observation_type_hash() != time2.observation_type_hash());
  CHECK(time0.hash() != time2.hash());

  CHECK(get_output(id0) ==
        std::string(MakeString{} << '(' << id0.observation_type_hash() << ","
                                 << id0.hash() << ',' << id0.value() << ')'));

  // Test PUP
  test_serialization(id0);
  test_serialization(id1);
  test_serialization(id2);
  test_serialization(id3);
  test_serialization(id4);
  test_serialization(time0);
  test_serialization(time1);
  test_serialization(time2);
}
}  // namespace observers
