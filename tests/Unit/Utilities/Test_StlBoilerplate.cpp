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

// [RandomAccessSequence]
template <typename T>
class SequenceView
    : public stl_boilerplate::RandomAccessSequence<SequenceView<T>, T, true> {
 public:
  SequenceView(const gsl::not_null<T*> data, const size_t size)
      : data_(data), size_(size) {}

  size_t size() const { return size_; }

  T& operator[](const size_t n) { return *(data_ + n); }
  const T& operator[](const size_t n) const { return *(data_ + n); }

 private:
  T* data_{nullptr};
  size_t size_{0};
};
// [RandomAccessSequence]

template <bool CheckConst>
void test_sequence_view() {
  using AccessType = tmpl::conditional_t<CheckConst, const int, int>;
  using View = SequenceView<AccessType>;

  static_assert(std::is_same_v<typename View::value_type, AccessType>);
  static_assert(std::is_same_v<typename View::reference, AccessType&>);
  static_assert(std::is_same_v<typename View::const_reference, const int&>);
  static_assert(std::is_same_v<typename View::pointer, AccessType*>);
  static_assert(std::is_same_v<typename View::const_pointer, const int*>);
  static_assert(std::is_same_v<typename View::difference_type, std::ptrdiff_t>);
  static_assert(std::is_same_v<typename View::size_type, size_t>);
  static_assert(std::is_convertible_v<typename View::iterator,
                                      typename View::const_iterator>);
  static_assert(
      std::is_convertible_v<typename std::iterator_traits<
                                typename View::iterator>::iterator_category,
                            std::random_access_iterator_tag>);
  static_assert(std::is_convertible_v<
                typename std::iterator_traits<
                    typename View::const_iterator>::iterator_category,
                std::random_access_iterator_tag>);

  std::array data{1, 2, 3, 4, 5};
  View view(&data[1], 3);
  View view_same(&data[1], 3);
  View view2(&data[1], 2);
  View view3(&data[0], 3);
  View view_empty(&data[1], 0);

  const View& cview = view;

  static_assert(
      std::is_same_v<decltype(view.begin()), typename View::iterator>);
  static_assert(std::is_same_v<decltype(view.end()), typename View::iterator>);
  static_assert(
      std::is_same_v<decltype(view.cbegin()), typename View::const_iterator>);
  static_assert(
      std::is_same_v<decltype(view.cend()), typename View::const_iterator>);
  static_assert(
      std::is_same_v<decltype(cview.begin()), typename View::const_iterator>);
  static_assert(
      std::is_same_v<decltype(cview.end()), typename View::const_iterator>);
  CHECK(*view.begin() == 2);
  CHECK(*(view.end() - 1) == 4);
  CHECK(*cview.begin() == 2);
  CHECK(*(cview.end() - 1) == 4);
  {
    typename View::iterator mi{};
    typename View::const_iterator ci{};
    mi = view.begin();
    ci = view.begin();
  }

  CHECK(cview == view_same);
  CHECK_FALSE(cview != view_same);
  CHECK_FALSE(cview == view2);
  CHECK(cview != view2);
  CHECK_FALSE(cview == view3);
  CHECK(cview != view3);
  CHECK_FALSE(cview == view_empty);
  CHECK(cview != view_empty);

  CHECK(cview.size() == 3);
  CHECK(view_empty.size() == 0);
  CHECK(not cview.empty());
  CHECK(view_empty.empty());
  CHECK(view.end() - view.begin() == static_cast<std::ptrdiff_t>(view.size()));
  CHECK(cview.end() - cview.begin() ==
        static_cast<std::ptrdiff_t>(view.size()));
  CHECK(view_empty.begin() == view_empty.end());
  CHECK(cview.max_size() >= view.size());

  static_assert(std::is_same_v<typename View::reverse_iterator,
                               std::reverse_iterator<typename View::iterator>>);
  static_assert(
      std::is_same_v<typename View::const_reverse_iterator,
                     std::reverse_iterator<typename View::const_iterator>>);
  static_assert(
      std::is_same_v<decltype(view.rbegin()), typename View::reverse_iterator>);
  static_assert(
      std::is_same_v<decltype(view.rend()), typename View::reverse_iterator>);
  static_assert(std::is_same_v<decltype(view.crbegin()),
                               typename View::const_reverse_iterator>);
  static_assert(std::is_same_v<decltype(view.crend()),
                               typename View::const_reverse_iterator>);
  static_assert(std::is_same_v<decltype(cview.rbegin()),
                               typename View::const_reverse_iterator>);
  static_assert(std::is_same_v<decltype(cview.rend()),
                               typename View::const_reverse_iterator>);
  CHECK(*view.rbegin() == 4);
  CHECK(*(view.rend() - 1) == 2);
  CHECK(view.rend() - view.rbegin() ==
        static_cast<std::ptrdiff_t>(view.size()));
  CHECK(*cview.rbegin() == 4);
  CHECK(*(cview.rend() - 1) == 2);
  CHECK(cview.rend() - cview.rbegin() ==
        static_cast<std::ptrdiff_t>(view.size()));

  CHECK_FALSE(cview < view_same);
  CHECK_FALSE(cview > view_same);
  CHECK(cview <= view_same);
  CHECK(cview >= view_same);
  CHECK_FALSE(cview < view2);
  CHECK_FALSE(view2 > cview);
  CHECK(cview > view2);
  CHECK(view2 < cview);
  CHECK_FALSE(cview <= view2);
  CHECK_FALSE(view2 >= cview);
  CHECK(cview >= view2);
  CHECK(view2 <= cview);
  CHECK_FALSE(cview < view3);
  CHECK_FALSE(view3 > cview);
  CHECK(cview > view3);
  CHECK(view3 < cview);
  CHECK_FALSE(cview <= view3);
  CHECK_FALSE(view3 >= cview);
  CHECK(cview >= view3);
  CHECK(view3 <= cview);
  CHECK_FALSE(cview < view_empty);
  CHECK_FALSE(view_empty > view3);
  CHECK(cview > view_empty);
  CHECK(view_empty < cview);
  CHECK_FALSE(cview <= view_empty);
  CHECK_FALSE(view_empty >= cview);
  CHECK(cview >= view_empty);
  CHECK(view_empty <= cview);

  static_assert(
      std::is_same_v<decltype(view.front()), decltype(*view.begin())>);
  CHECK(&view.front() == &*view.begin());
  static_assert(
      std::is_same_v<decltype(cview.front()), decltype(*view.cbegin())>);
  CHECK(&cview.front() == &*view.cbegin());
  static_assert(
      std::is_same_v<decltype(view.back()), decltype(*view.rbegin())>);
  CHECK(&view.back() == &*view.rbegin());
  static_assert(
      std::is_same_v<decltype(cview.back()), decltype(*view.crbegin())>);
  CHECK(&cview.back() == &*view.crbegin());

  CHECK(view[1] == 3);
  CHECK(&view[1] == &data[2]);
  CHECK(view.at(1) == 3);
  CHECK(&view.at(1) == &data[2]);
  CHECK_THROWS_AS(view.at(view.size()), std::out_of_range);
  CHECK(cview.at(1) == 3);
  CHECK(&cview.at(1) == &data[2]);
  CHECK_THROWS_AS(cview.at(view.size()), std::out_of_range);

  if constexpr (not CheckConst) {
    CHECK(&(view[1] = 12) == &data[2]);
    CHECK(&(view.at(1) = 12) == &data[2]);
    CHECK(&(view.front() = 12) == &data[1]);
    CHECK(&(view.back() = 12) == &data[3]);
  }
}

void test_random_access_sequence() {
  test_sequence_view<false>();
  test_sequence_view<true>();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.StlBoilerplate", "[Unit][Utilities]") {
  test_random_access_iterator();
  test_random_access_sequence();
}
