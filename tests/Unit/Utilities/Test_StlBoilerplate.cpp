// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <vector>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StlBoilerplate.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// Iterate over two std::vectors.
template <typename T>
class TwoVectorIterator
    : public stl_boilerplate::RandomAccessIterator<TwoVectorIterator<T>, T> {
  // In a real use would set constness of Vector correctly.
  using Vector = std::vector<std::remove_const_t<T>>;

 public:
  TwoVectorIterator() = default;
  TwoVectorIterator(const gsl::not_null<Vector*> v1,
                    const gsl::not_null<Vector*> v2)
      : v1_(v1), v2_(v2) {}

  T& operator*() const {
    if (offset_ < v1_->size()) {
      return (*v1_)[offset_];
    } else {
      return (*v2_)[offset_ - v1_->size()];
    }
  }

  TwoVectorIterator& operator+=(const std::ptrdiff_t n) {
    offset_ += static_cast<size_t>(n);
    return *this;
  }

  friend inline bool operator==(const TwoVectorIterator& a,
                                const TwoVectorIterator& b) {
    return a.v1_ == b.v1_ and a.v2_ == b.v2_ and a.offset_ == b.offset_;
  }

  friend inline std::ptrdiff_t operator-(const TwoVectorIterator& a,
                                         const TwoVectorIterator& b) {
    return static_cast<std::ptrdiff_t>(a.offset_) -
           static_cast<std::ptrdiff_t>(b.offset_);
  }

 private:
  Vector* v1_{nullptr};
  Vector* v2_{nullptr};
  size_t offset_{0};
};

// [RandomAccessIterator]
template <typename T>
class EmptyIterator
    : public stl_boilerplate::RandomAccessIterator<EmptyIterator<T>, T> {
 public:
  T& operator*() const { ERROR("Not dereferenceable."); }

  EmptyIterator& operator+=(const std::ptrdiff_t /*n*/) { return *this; }
};

// Often these will be friend functions.
template <typename T>
bool operator==(const EmptyIterator<T>& /*a*/, const EmptyIterator<T>& /*b*/) {
  return true;
}

template <typename T>
std::ptrdiff_t operator-(const EmptyIterator<T>& /*a*/,
                         const EmptyIterator<T>& /*b*/) {
  return 0;
}
// [RandomAccessIterator]

using empty_traits = std::iterator_traits<EmptyIterator<char>>;
static_assert(std::is_same_v<empty_traits::iterator_category,
                             std::random_access_iterator_tag>);
static_assert(std::is_same_v<empty_traits::value_type, char>);
static_assert(std::is_same_v<empty_traits::difference_type, std::ptrdiff_t>);
static_assert(std::is_same_v<empty_traits::reference, char&>);
static_assert(std::is_same_v<empty_traits::pointer, char*>);

template <bool CheckConst>
void test_two_vector_iterator() {
  using Iterator =
      TwoVectorIterator<tmpl::conditional_t<CheckConst, const int, int>>;
  using traits = std::iterator_traits<Iterator>;
  static_assert(std::is_same_v<typename traits::iterator_category,
                               std::random_access_iterator_tag>);
  static_assert(std::is_same_v<typename traits::value_type, int>);
  static_assert(
      std::is_same_v<typename traits::difference_type, std::ptrdiff_t>);
  if constexpr (CheckConst) {
    static_assert(std::is_same_v<typename traits::reference, const int&>);
    static_assert(std::is_same_v<typename traits::pointer, const int*>);
  } else {
    static_assert(std::is_same_v<typename traits::reference, int&>);
    static_assert(std::is_same_v<typename traits::pointer, int*>);
  }

  std::vector<int> v1{1, 2};
  std::vector<int> v2{11, 12};
  Iterator it(&v1, &v2);
  Iterator it2 = it;
  CHECK(*it == 1);
  {
    Iterator& incremented = ++it;
    CHECK(&incremented == &it);
  }
  CHECK(*it == 2);

  // input
  CHECK(it.operator->() == &v1[1]);
  CHECK_FALSE(it == it2);
  CHECK(it != it2);
  ++it2;
  CHECK(it == it2);
  CHECK_FALSE(it != it2);
  CHECK(*it++ == 2);
  CHECK(*it == 11);

  // output
  if constexpr (not CheckConst) {
    *it = 21;
    CHECK(*it == 21);
  }

  // forward
  CHECK(Iterator{} == Iterator{});

  // bidirectional
  {
    Iterator& decremented = --it;
    CHECK(&decremented == &it);
  }
  CHECK(*it == 2);
  CHECK(it == it2);
  CHECK(*it-- == 2);
  CHECK(*it == 1);

  // random access
  {
    decltype(auto) ret = it += 3;
    CHECK(&ret == &it);
  }
  CHECK(*it == 12);
  {
    decltype(auto) ret = it2 + 2;
    CHECK(&ret != &it2);
    CHECK(ret != it2);
    CHECK(ret == it);
  }
  CHECK(*it2 == 2);
  {
    decltype(auto) ret = 2 + it2;
    CHECK(&ret != &it2);
    CHECK(ret != it2);
    CHECK(ret == it);
  }
  CHECK(*it2 == 2);
  {
    decltype(auto) ret = it + (-2);
    CHECK(&ret != &it);
    CHECK(ret != it);
    CHECK(ret == it2);
  }
  CHECK(*it == 12);
  {
    decltype(auto) ret = (-2) + it;
    CHECK(&ret != &it);
    CHECK(ret != it);
    CHECK(ret == it2);
  }
  CHECK(*it == 12);
  {
    decltype(auto) ret = it - 2;
    CHECK(&ret != &it);
    CHECK(ret != it);
    CHECK(ret == it2);
  }
  CHECK(*it == 12);
  {
    decltype(auto) ret = it2 - (-2);
    CHECK(&ret != &it2);
    CHECK(ret != it2);
    CHECK(ret == it);
  }
  CHECK(*it2 == 2);
  it -= 2;
  CHECK(it == it2);
  it -= -2;
  CHECK(*it == 12);
  it += -2;
  CHECK(it == it2);

  CHECK(it - it2 == 0);
  it += 2;
  CHECK(it - it2 == 2);
  CHECK(it2 - it == -2);

  CHECK(it2[0] == 2);
  CHECK(&it2[0] == &v1[1]);
  CHECK(it2[2] == 12);
  CHECK(it2[-1] == 1);
  if constexpr (not CheckConst) {
    it2[2] = 30;
    CHECK(*it == 30);
  }

  CHECK(it > it2);
  CHECK(it >= it2);
  CHECK_FALSE(it < it2);
  CHECK_FALSE(it <= it2);
  CHECK_FALSE(it2 > it);
  CHECK_FALSE(it2 >= it);
  CHECK(it2 < it);
  CHECK(it2 <= it);
  CHECK_FALSE(it < it);
  CHECK_FALSE(it > it);
  CHECK(it <= it);
  CHECK(it >= it);
}

void test_random_access_iterator() {
  test_two_vector_iterator<false>();
  test_two_vector_iterator<true>();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.StlBoilerplate", "[Unit][Utilities]") {
  test_random_access_iterator();
}
