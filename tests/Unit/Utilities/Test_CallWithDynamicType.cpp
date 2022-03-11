// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <utility>

#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.call_with_dynamic_type",
                  "[Unit][Utilities]") {
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

  CHECK(1 ==
        (call_with_dynamic_type<int, tmpl::list<D1, D2>>(
             d1.get(), [](auto* const p) { return p->mutable_func(); })));
  CHECK(2 ==
        (call_with_dynamic_type<int, tmpl::list<D1, D2>>(
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
