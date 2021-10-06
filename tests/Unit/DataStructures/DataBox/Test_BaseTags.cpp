// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Subitems.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace TestTags {
// [vector_base_definitions]
template <int I>
struct VectorBase : db::BaseTag {};

template <int I>
struct Vector : db::SimpleTag, VectorBase<I> {
  using type = std::vector<double>;
};
// [vector_base_definitions]

// [array_base_definitions]
template <int I>
struct ArrayBase : db::BaseTag {};

template <int I, size_t Size = 3>
struct Array : db::SimpleTag, ArrayBase<I> {
  using type = std::array<int, Size>;
};
// [array_base_definitions]

// [compute_template_base_tags]
template <int I, int VectorBaseIndex = 0, int... VectorBaseExtraIndices>
struct ArrayCompute : Array<I, sizeof...(VectorBaseExtraIndices) == 0
                                   ? 3
                                   : 2 + sizeof...(VectorBaseExtraIndices)>,
                      db::ComputeTag {
  using base = Array<I, sizeof...(VectorBaseExtraIndices) == 0
                            ? 3
                            : 2 + sizeof...(VectorBaseExtraIndices)>;
  using return_type = typename base::type;

  static void function(const gsl::not_null<std::array<int, 3>*> result,
                       const std::vector<double>& t) {
    *result = {{static_cast<int>(t.size()), static_cast<int>(t[0]), -8}};
  }

  template <typename... Args>
  static void function(
      const gsl::not_null<std::array<int, 2 + sizeof...(Args)>*> result,
      const std::vector<double>& t, const Args&... args) {
    static_assert(sizeof...(VectorBaseExtraIndices) == sizeof...(Args));
    *result = {{static_cast<int>(t.size()), static_cast<int>(t[0]),
                static_cast<int>(args[0])...}};
  }

  using argument_tags = tmpl::list<VectorBase<VectorBaseIndex>,
                                   VectorBase<VectorBaseExtraIndices>...>;
};
// [compute_template_base_tags]
}  // namespace TestTags

void test_non_subitems() {
  // Test "easy case" where there are no subitems. Test:
  // - `get`ing an item by a base tag
  // - `mutate`ing an item by a base tag
  // - using a base tag as the argument in a compute item
  // - `get`ing a compute item by base tag
  // - `get`ing a compute item by its simple tag
  // [base_simple_and_compute_mutate]
  auto box = db::create<db::AddSimpleTags<TestTags::Vector<0>>,
                        db::AddComputeTags<TestTags::ArrayCompute<0>>>(
      std::vector<double>{-10.0, 10.0});

  // Check retrieving simple tag Vector<0> using base tag VectorBase<0>
  CHECK(db::get<TestTags::VectorBase<0>>(box) ==
        std::vector<double>{-10.0, 10.0});

  // Check retrieving compute tag ArrayCompute<0> using simple tag Array<0>
  CHECK(db::get<TestTags::Array<0>>(box) == std::array<int, 3>{{2, -10, -8}});

  // Check mutating Vector<0> using VectorBase<0>
  db::mutate<TestTags::VectorBase<0>>(
      make_not_null(&box), [](const auto vector) { (*vector)[0] = 101.8; });

  CHECK(db::get<TestTags::VectorBase<0>>(box) ==
        std::vector<double>{101.8, 10.0});

  // Check retrieving ArrayCompute<0> using base tag ArrayBase<0>.
  // ArrayCompute was reset after mutating Vector<0>
  CHECK(db::get<TestTags::ArrayBase<0>>(box) ==
        std::array<int, 3>{{2, 101, -8}});

  // Check retrieving ArrayCompute<0> using simple tag Array<0>.
  CHECK(db::get<TestTags::Array<0>>(box) == std::array<int, 3>{{2, 101, -8}});
  CHECK(db::get<TestTags::ArrayCompute<0>>(box) ==
        std::array<int, 3>{{2, 101, -8}});
  // [base_simple_and_compute_mutate]

  // - adding compute item that uses a base tag as its argument
  auto box2 = db::create_from<db::RemoveTags<>, db::AddSimpleTags<>,
                              db::AddComputeTags<TestTags::ArrayCompute<1>>>(
      std::move(box));
  CHECK(db::get<TestTags::ArrayBase<1>>(box2) ==
        std::array<int, 3>{{2, 101, -8}});
  CHECK(db::get<TestTags::ArrayCompute<1>>(box2) ==
        std::array<int, 3>{{2, 101, -8}});
  const auto box3 = db::create_from<db::RemoveTags<TestTags::ArrayCompute<1>>>(
      std::move(box2));
  CHECK(db::get<TestTags::VectorBase<0>>(box3) ==
        std::vector<double>{101.8, 10.0});
  CHECK(db::get<TestTags::ArrayBase<0>>(box3) ==
        std::array<int, 3>{{2, 101, -8}});
  CHECK(db::get<TestTags::ArrayCompute<0>>(box3) ==
        std::array<int, 3>{{2, 101, -8}});

  // - adding a new simple item and a compute item that uses it and an
  //   existing item via a base tag. This is implemented as a function
  //   template compute item to also test that compute items can be function
  //   templates.
  // - mutate one of the arguments that are template parameters of the compute
  //   item and also a base tag
  auto box4 = db::create_from<
      db::RemoveTags<>,
      db::AddSimpleTags<TestTags::Vector<1>, TestTags::Vector<2>>,
      db::AddComputeTags<TestTags::ArrayCompute<1, 0, 1, 2>>>(
      db::create<db::AddSimpleTags<TestTags::Vector<0>>,
                 db::AddComputeTags<TestTags::ArrayCompute<0>>>(
          std::vector<double>{101.8, 10.0}),
      std::vector<double>{-7.1, 8.9}, std::vector<double>{103.1, -73.2});
  CHECK(db::get<TestTags::ArrayBase<1>>(box4) ==
        std::array<int, 4>{{2, 101, -7, 103}});
  db::mutate<TestTags::VectorBase<2>>(
      make_not_null(&box4), [](const auto vector) { (*vector)[0] = 408.8; });
  CHECK(db::get<TestTags::ArrayBase<1>>(box4) ==
        std::array<int, 4>{{2, 101, -7, 408}});

  // - removing a compute item that uses a base tag as its argument
  // - removing a compute item by its base tag
  const auto box5 =
      db::create_from<db::RemoveTags<TestTags::ArrayBase<1>>>(std::move(box4));
  CHECK(db::get<TestTags::VectorBase<1>>(box5) ==
        std::vector<double>{-7.1, 8.9});
  CHECK(db::get<TestTags::VectorBase<2>>(box5) ==
        std::vector<double>{408.8, -73.2});

  // - removing a compute item and its dependencies by the base tags
  // [remove_using_base]
  const auto& box6 = db::create_from<
      db::RemoveTags<TestTags::VectorBase<1>, TestTags::VectorBase<2>,
                     TestTags::ArrayBase<1>>>(
      db::create_from<
          db::RemoveTags<>,
          db::AddSimpleTags<TestTags::Vector<1>, TestTags::Vector<2>>,
          db::AddComputeTags<TestTags::ArrayCompute<1, 0, 1, 2>>>(
          db::create<db::AddSimpleTags<TestTags::Vector<0>>,
                     db::AddComputeTags<TestTags::ArrayCompute<0>>>(
              std::vector<double>{101.8, 10.0}),
          std::vector<double>{-7.1, 8.9}, std::vector<double>{408.8, -73.2}));
  // [remove_using_base]
  CHECK(db::get<TestTags::VectorBase<0>>(box6) ==
        std::vector<double>{101.8, 10.0});
  CHECK(db::get<TestTags::ArrayBase<0>>(box6) ==
        std::array<int, 3>{{2, 101, -8}});
}

namespace TestTags {
// We can't use raw fundamental types as subitems because subitems
// need to have a reference-like nature.
template <typename T>
class Boxed {
 public:
  explicit Boxed(std::shared_ptr<T> data) : data_(std::move(data)) {}
  Boxed() = default;
  // The multiple copy constructors (assignment operators) are needed
  // to prevent users from modifying compute item values.
  Boxed(const Boxed&) = delete;
  Boxed(Boxed&) = default;
  Boxed(Boxed&&) = default;
  Boxed& operator=(const Boxed&) = delete;
  Boxed& operator=(Boxed&) = default;
  Boxed& operator=(Boxed&&) = default;
  ~Boxed() = default;

  T& operator*() { return *data_; }
  const T& operator*() const { return *data_; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) {
    if (p.isUnpacking()) {
      T t{};
      p | t;
      data_ = std::make_shared<T>(std::move(t));
    } else {
      p | *data_;
    }
  }

 private:
  std::shared_ptr<T> data_;
};

template <typename T>
bool operator==(const Boxed<T>& lhs, const Boxed<T>& rhs) {
  return *lhs == *rhs;
}

template <typename T>
bool operator!=(const Boxed<T>& lhs, const Boxed<T>& rhs) {
  return not(*lhs == *rhs);
}

template <size_t N>
struct FirstBase : db::BaseTag {};

template <size_t N>
struct SecondBase : db::BaseTag {};

template <size_t N>
struct ParentBase : db::BaseTag {};

template <size_t N>
struct Parent : ParentBase<N>, db::SimpleTag {
  using type = std::pair<Boxed<int>, Boxed<double>>;
};

template <size_t N>
struct ParentCompute : Parent<N>, db::ComputeTag {
  using base = Parent<N>;
  using return_type = std::pair<Boxed<int>, Boxed<double>>;
  static void function(const gsl::not_null<return_type*> result,
                       const std::pair<Boxed<int>, Boxed<double>>& arg) {
    *result = std::make_pair(
        Boxed<int>(std::make_shared<int>(*arg.first + 1)),
        Boxed<double>(std::make_shared<double>(*arg.second * 2.)));
  }
  using argument_tags = tmpl::list<ParentBase<N - 1>>;
};

template <size_t N>
struct First : FirstBase<N>, db::SimpleTag {
  using type = Boxed<int>;

  static constexpr size_t index = 0;
};
template <size_t N>
struct Second : SecondBase<N>, db::SimpleTag {
  using type = Boxed<double>;

  static constexpr size_t index = 1;
};

template <size_t N0, size_t N1>
struct MultiplyByTwoBase : db::BaseTag {};

template <size_t N0, size_t N1>
struct MultiplyByTwo : MultiplyByTwoBase<N0, N1>, db::SimpleTag {
  using type = double;
};

template <size_t N0, size_t N1>
struct MultiplyByTwoCompute : MultiplyByTwo<N0, N1>, db::ComputeTag {
  using base = MultiplyByTwo<N0, N1>;
  using return_type = double;
  // We use a function template and auto return type solely to test that these
  // work correctly with the DataBox.
  template <typename T0, typename T1>
  static void function(const gsl::not_null<double*> result, const T0& t0,
                       const T1& t1) {
    *result = *t0 * *t1;
  }
  using argument_tags = tmpl::list<First<N0>, Second<N1>>;
};
}  // namespace TestTags
}  // namespace

namespace db {
template <size_t N>
struct Subitems<TestTags::Parent<N>> {
  using type = tmpl::list<TestTags::First<N>, TestTags::Second<N>>;
  using tag = TestTags::Parent<N>;

  // The argument types to create_item are template parameters instead of
  // gsl::not_null<item_type<tag>*> and gsl::not_null<item_type<Subtag>*> to
  // test that the code works correctly if create_item is a function template
  template <typename Subtag, typename T0, typename T1>
  static void create_item(const T0 parent_value, const T1 sub_value) {
    *sub_value = std::get<Subtag::index>(*parent_value);
  }

  // We use the template parameter T instead of item_type<tag, TagList> just to
  // test that create_compute_time functions can be function templates too
  template <typename Subtag, typename T>
  static const auto& create_compute_item(const T& parent_value) {
    return std::get<Subtag::index>(parent_value);
  }
};

template <size_t N>
struct Subitems<TestTags::ParentCompute<N>> {
  using type = tmpl::list<TestTags::First<N>, TestTags::Second<N>>;
  using tag = TestTags::ParentCompute<N>;

  // The argument types to create_item are template parameters instead of
  // gsl::not_null<item_type<tag>*> and gsl::not_null<item_type<Subtag>*> to
  // test that the code works correctly if create_item is a function template
  template <typename Subtag, typename T0, typename T1>
  static void create_item(const T0 parent_value, const T1 sub_value) {
    *sub_value = std::get<Subtag::index>(*parent_value);
  }

  // We use the template parameter T instead of item_type<tag, TagList> just to
  // test that create_compute_time functions can be function templates too
  template <typename Subtag, typename T>
  static const auto& create_compute_item(const T& parent_value) {
    return std::get<Subtag::index>(parent_value);
  }
};
}  // namespace db

namespace {
void test_subitems_tags() {
  // Test subitems cases:
  // - `get`ing a subitem by a base tag
  // - `get`ing a subitem of a compute item by base tag
  auto box = db::create<db::AddSimpleTags<TestTags::Parent<0>>,
                        db::AddComputeTags<TestTags::ParentCompute<1>>>(
      std::make_pair(TestTags::Boxed<int>(std::make_shared<int>(5)),
                     TestTags::Boxed<double>(std::make_shared<double>(3.5))));
  CHECK(*db::get<TestTags::FirstBase<0>>(box) == 5);
  CHECK(*db::get<TestTags::SecondBase<0>>(box) == 3.5);
  CHECK(db::get<TestTags::ParentBase<0>>(box) ==
        std::make_pair(TestTags::Boxed<int>(std::make_shared<int>(5)),
                       TestTags::Boxed<double>(std::make_shared<double>(3.5))));
  CHECK(*db::get<TestTags::FirstBase<1>>(box) == 6);
  CHECK(*db::get<TestTags::SecondBase<1>>(box) == 7.0);
  CHECK(db::get<TestTags::ParentBase<1>>(box) ==
        std::make_pair(TestTags::Boxed<int>(std::make_shared<int>(6)),
                       TestTags::Boxed<double>(std::make_shared<double>(7.0))));

  // - `mutate`ing a subitem by a base tag
  db::mutate<TestTags::FirstBase<0>>(
      make_not_null(&box),
      [](const gsl::not_null<TestTags::Boxed<int>*> first) { **first = -3; });
  CHECK(*db::get<TestTags::FirstBase<0>>(box) == -3);
  CHECK(*db::get<TestTags::SecondBase<0>>(box) == 3.5);
  CHECK(db::get<TestTags::ParentBase<0>>(box) ==
        std::make_pair(TestTags::Boxed<int>(std::make_shared<int>(-3)),
                       TestTags::Boxed<double>(std::make_shared<double>(3.5))));
  CHECK(*db::get<TestTags::FirstBase<1>>(box) == -2);
  CHECK(*db::get<TestTags::SecondBase<1>>(box) == 7.0);
  CHECK(db::get<TestTags::ParentBase<1>>(box) ==
        std::make_pair(TestTags::Boxed<int>(std::make_shared<int>(-2)),
                       TestTags::Boxed<double>(std::make_shared<double>(7.0))));

  // - adding a compute item that uses subitems via base tags, one that's
  //   already in the box another that's being added
  const auto box2 =
      db::create_from<db::RemoveTags<>, db::AddSimpleTags<>,
                      db::AddComputeTags<TestTags::MultiplyByTwoCompute<0, 1>>>(
          std::move(box));
  CHECK(db::get<TestTags::MultiplyByTwoBase<0, 1>>(box2) == -3 * 7.0);

  // - using a subitem by base tag as the argument to a compute item (make
  //   sure mutating works correctly, both if mutating the base and if
  //   mutating the collection)
  // - Test mutating the Subitem itself rather than one of the subitems.
  auto box3 =
      db::create_from<db::RemoveTags<>, db::AddSimpleTags<TestTags::Parent<2>>,
                      db::AddComputeTags<TestTags::MultiplyByTwoCompute<0, 2>>>(
          db::create<db::AddSimpleTags<TestTags::Parent<0>>,
                     db::AddComputeTags<TestTags::ParentCompute<1>>>(
              std::make_pair(
                  TestTags::Boxed<int>(std::make_shared<int>(-3)),
                  TestTags::Boxed<double>(std::make_shared<double>(3.5)))),
          std::make_pair(
              TestTags::Boxed<int>(std::make_shared<int>(-3)),
              TestTags::Boxed<double>(std::make_shared<double>(9.5))));
  CHECK(db::get<TestTags::MultiplyByTwoBase<0, 2>>(box3) == -3 * 9.5);
  db::mutate<TestTags::FirstBase<0>>(
      make_not_null(&box3),
      [](const gsl::not_null<TestTags::Boxed<int>*> first) { **first = 4; });
  CHECK(db::get<TestTags::MultiplyByTwoBase<0, 2>>(box3) == 4 * 9.5);
  db::mutate<TestTags::ParentBase<0>>(
      make_not_null(&box3),
      [](const gsl::not_null<
          std::pair<TestTags::Boxed<int>, TestTags::Boxed<double>>*>
             parent0) { *parent0->first = 8; });
  CHECK(db::get<TestTags::MultiplyByTwoBase<0, 2>>(box3) == 8 * 9.5);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.BaseTags",
                  "[Unit][DataStructures]") {
  test_non_subitems();
  test_subitems_tags();
}
