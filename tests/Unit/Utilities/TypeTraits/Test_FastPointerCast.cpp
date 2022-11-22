// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>

#include "Utilities/TypeTraits/FastPointerCast.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.TypeTraits.FastPointerCast",
                  "[Unit][Utilities]") {
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

  CHECK(1 == (tt::fast_pointer_cast<D1* const>(d1.get())->mutable_func()));
  CHECK(2 == (tt::fast_pointer_cast<D2* const>(d2.get())->mutable_func()));
  CHECK(3 == (tt::fast_pointer_cast<C* const>(c.get())->mutable_func()));

  CHECK(11 == (tt::fast_pointer_cast<const D1* const>(&std::as_const(*d1))
                   ->const_func()));
  CHECK(12 == (tt::fast_pointer_cast<const D2* const>(&std::as_const(*d2))
                   ->const_func()));
  CHECK(13 == (tt::fast_pointer_cast<const C* const>(&std::as_const(*c))
                   ->const_func()));
}
