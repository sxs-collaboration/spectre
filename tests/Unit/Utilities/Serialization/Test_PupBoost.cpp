// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include <boost/math/quaternion.hpp>
#include <boost/rational.hpp>
#include <cstddef>
#include <functional>
#include <type_traits>

#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupBoost.hpp"
#include "Utilities/Serialization/Serialize.hpp"

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

// Boost 1.86 added an undocumented `is_small()` method, but we have
// to support older versions.
template <typename T, size_t N>
bool small_vector_is_small(const boost::container::small_vector<T, N>& v) {
  // The pointer comparison rules are weird and std::less is not
  // equivalent to operator<.
  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast, cppcoreguidelines-pro-bounds-pointer-arithmetic)
  return std::less_equal<const char*>{}(
             reinterpret_cast<const char*>(&v),
             reinterpret_cast<const char*>(v.data())) and
         std::less<const char*>{}(reinterpret_cast<const char*>(v.data()),
                                  reinterpret_cast<const char*>(&v + 1));
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast, cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

template <typename T>
void test_small_vector() {
  boost::container::small_vector<T, 5> small_vector{};
  for (int i = 0; i < 3; ++i) {
    small_vector.emplace_back(i);
  }
  boost::container::small_vector<T, 5> large_vector{};
  for (int i = 0; i < 6; ++i) {
    large_vector.emplace_back(i);
  }

  // Test that the function works
  REQUIRE(small_vector_is_small(small_vector));
  REQUIRE(not small_vector_is_small(large_vector));

  boost::container::small_vector<T, 5> copied_small_vector{};
  boost::container::small_vector<T, 5> copied_large_vector{};
  serialize_and_deserialize(make_not_null(&copied_small_vector), small_vector);
  serialize_and_deserialize(make_not_null(&copied_large_vector), large_vector);

  CHECK(copied_small_vector == small_vector);
  CHECK(copied_large_vector == large_vector);
  CHECK(small_vector_is_small(copied_small_vector));
  CHECK(not small_vector_is_small(copied_large_vector));
}

void test_quaternion() {
  boost::math::quaternion<double> quat(2.0, 3.0, 4.0, 5.0);
  test_serialization(quat);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Serialization.PupBoost", "[Unit][Serialization]") {
  test_rational();
  test_static_vector();
  test_small_vector<int>();
  test_small_vector<NotTriviallyCopyable>();
  test_quaternion();
}
