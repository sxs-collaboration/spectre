// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/container/static_vector.hpp>
#include <boost/rational.hpp>
#include <cstddef>
#include <type_traits>

#include "Framework/TestHelpers.hpp"
#include "Utilities/Serialization/PupBoost.hpp"

namespace {
void test_rational() {
  boost::rational<size_t> r1(3_st, 4_st);
  test_serialization(r1);
  boost::rational<int> r2(-5, 2);
  test_serialization(r2);
}

struct NotTriviallyCopyable {
  NotTriviallyCopyable() = default;
  NotTriviallyCopyable(const NotTriviallyCopyable& other) : data(other.data) {}
  NotTriviallyCopyable& operator=(const NotTriviallyCopyable& other) {
    data = other.data;
    return *this;
  }
  explicit NotTriviallyCopyable(const int d) : data(d) {}

  int data;

  void pup(PUP::er& p) { p | data; }
};

static_assert(not std::is_trivially_copyable_v<NotTriviallyCopyable>);

bool operator==(const NotTriviallyCopyable& a, const NotTriviallyCopyable& b) {
  return a.data == b.data;
}

void test_static_vector() {
  boost::container::static_vector<int, 5> vector_int{1, 2, 3};
  test_serialization(vector_int);
  boost::container::static_vector<NotTriviallyCopyable, 5> vector_ntc{};
  vector_ntc.emplace_back(1);
  vector_ntc.emplace_back(2);
  vector_ntc.emplace_back(3);
  test_serialization(vector_ntc);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Serialization.PupBoost", "[Unit][Serialization]") {
  test_rational();
  test_static_vector();
}
