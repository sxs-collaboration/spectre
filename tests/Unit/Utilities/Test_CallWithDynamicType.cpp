// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <utility>

#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/TMPL.hpp"

namespace {
void test_static_cast() {
  struct B {
    B() = default;
    B(const B&) = delete;
    B(B&&) = delete;
    B& operator=(const B&) = delete;
    B& operator=(B&&) = delete;
    virtual ~B() = default;
  };
  struct D1 : B {
    int mutable_func() { return 1; }
    int const_func() const { return 11; }
  };
  struct D2 : B {
    int mutable_func() { return 2; }
    int const_func() const { return 12; }
  };
  std::unique_ptr<B> d1 = std::make_unique<D1>();
  std::unique_ptr<B> d2 = std::make_unique<D2>();

  CHECK(1 == (call_with_dynamic_type<int, tmpl::list<D1, D2>>(
                 d1.get(), [](auto* const p) { return p->mutable_func(); })));
  CHECK(2 == (call_with_dynamic_type<int, tmpl::list<D1, D2>>(
                 d2.get(), [](auto* const p) { return p->mutable_func(); })));

  CHECK(11 == (call_with_dynamic_type<int, tmpl::list<D1, D2>>(
                  &std::as_const(*d1),
                  [](const auto* const p) { return p->const_func(); })));
  CHECK(12 == (call_with_dynamic_type<int, tmpl::list<D1, D2>>(
                  &std::as_const(*d2),
                  [](const auto* const p) { return p->const_func(); })));

  // Test void return type
  int result = 0;
  call_with_dynamic_type<void, tmpl::list<D1, D2>>(
      d2.get(), [&result](auto* const p) { result = p->mutable_func(); });
  CHECK(2 == result);
}

void test_dynamic_cast() {
  struct B {
    B() = default;
    B(const B&) = delete;
    B(B&&) = delete;
    B& operator=(const B&) = delete;
    B& operator=(B&&) = delete;
    virtual ~B() = default;
  };
  struct D1 : virtual B {
    int mutable_func() { return 1; }
    int const_func() const { return 11; }
  };
  struct D2 : virtual B {
    int mutable_func() { return 2; }
    int const_func() const { return 12; }
  };
  struct C : D1, D2 {
    int mutable_func() { return 3; }
    int const_func() const { return 13; }
  };
  std::unique_ptr<B> d1 = std::make_unique<D1>();
  std::unique_ptr<B> d2 = std::make_unique<D2>();
  std::unique_ptr<B> c = std::make_unique<C>();

  CHECK(1 == (call_with_dynamic_type<int, tmpl::list<D1, D2, C>>(
                 d1.get(), [](auto* const p) { return p->mutable_func(); })));
  CHECK(2 == (call_with_dynamic_type<int, tmpl::list<D1, D2, C>>(
                 d2.get(), [](auto* const p) { return p->mutable_func(); })));
  CHECK(3 == (call_with_dynamic_type<int, tmpl::list<D1, D2, C>>(
                 c.get(), [](auto* const p) { return p->mutable_func(); })));

  CHECK(11 == (call_with_dynamic_type<int, tmpl::list<D1, D2, C>>(
                  &std::as_const(*d1),
                  [](const auto* const p) { return p->const_func(); })));
  CHECK(12 == (call_with_dynamic_type<int, tmpl::list<D1, D2, C>>(
                  &std::as_const(*d2),
                  [](const auto* const p) { return p->const_func(); })));
  CHECK(13 == (call_with_dynamic_type<int, tmpl::list<D1, D2, C>>(
                  &std::as_const(*c),
                  [](const auto* const p) { return p->const_func(); })));

  // Test void return type
  int result = 0;
  call_with_dynamic_type<void, tmpl::list<D1, D2, C>>(
      d2.get(), [&result](auto* const p) { result = p->mutable_func(); });
  CHECK(2 == result);
  call_with_dynamic_type<void, tmpl::list<D1, D2, C>>(
      c.get(), [&result](auto* const p) { result = p->mutable_func(); });
  CHECK(3 == result);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.call_with_dynamic_type",
                  "[Unit][Utilities]") {
  test_static_cast();
  test_dynamic_cast();
}
