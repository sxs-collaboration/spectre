// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <memory>
#include <string>
#include <vector>

#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace {

/// [fake_virtual_example]
DEFINE_FAKE_VIRTUAL(fv)

class Derived;

class Base {
 public:
  using Inherit = FakeVirtualInherit_fv<Base>;

 protected:
  Base() = default;
  Base(const Base&) = default;
  Base(Base&&) = default;
  Base& operator=(const Base&) = default;
  Base& operator=(Base&&) = default;

 public:
  virtual ~Base() = default;

  template <typename T>
  int fv(int x) {
    return fake_virtual_fv<tmpl::list<Derived>, T>(this, x);
  }
};

class Derived : public Base::Inherit {
 public:
  template <typename T>
  int fv(int x) {
    return x + 3;
  }
};
/// [fake_virtual_example]

std::string called_func;
std::vector<std::string> called_types;

template <typename... T>
std::vector<std::string> make_type_vector() {
  return {pretty_type::get_name<T>()...};
}

DEFINE_FAKE_VIRTUAL(deduced)
DEFINE_FAKE_VIRTUAL(nondeduced)
DEFINE_FAKE_VIRTUAL(deduced_and_nondeduced)

class A;
class B;
using AB = tmpl::list<A, B>;

class MultipleBase {
 public:
  using Inherit = FakeVirtualInherit_deduced<FakeVirtualInherit_nondeduced<
      FakeVirtualInherit_deduced_and_nondeduced<MultipleBase>>>;

 protected:
  MultipleBase() = default;
  MultipleBase(const MultipleBase&) = default;
  MultipleBase(MultipleBase&&) = default;
  MultipleBase& operator=(const MultipleBase&) = default;
  MultipleBase& operator=(MultipleBase&&) = default;

 public:
  virtual ~MultipleBase() = default;

  template <typename T>
  T deduced(const T& t) /* not const for testing */ {
    return fake_virtual_deduced<AB>(this, t);
  }

  template <typename T>
  // clang-tidy: non-const reference argument - allows single test to
  // check reference args and mutable reference return type
  int& nondeduced(int& x) const {  // NOLINT
    return fake_virtual_nondeduced<AB, T>(this, x);
  }

  template <typename T, typename U>
  void deduced_and_nondeduced(const U& u) const {
    fake_virtual_deduced_and_nondeduced<AB, T>(this, u);
  }
};

class A : public MultipleBase::Inherit {
 public:
  template <typename T>
  T deduced(const T& t) /* not const for testing */ {
    called_func = "A::deduced";
    called_types = make_type_vector<T>();
    return t;
  }

  template <typename T>
  // clang-tidy: see base class
  int& nondeduced(int& x) const {  // NOLINT
    called_func = "A::nondeduced";
    called_types = make_type_vector<T>();
    return x;
  }

  template <typename T, typename U>
  void deduced_and_nondeduced(const U& /*unused*/) const {
    called_func = "A::deduced_and_nondeduced";
    called_types = make_type_vector<T, U>();
  }
};

class B : public MultipleBase::Inherit {
 public:
  template <typename T>
  T deduced(const T& t) /* not const for testing */ {
    called_func = "B::deduced";
    called_types = make_type_vector<T>();
    return t;
  }

  template <typename T>
  // clang-tidy: see base class
  int& nondeduced(int& x) const {  // NOLINT
    called_func = "B::nondeduced";
    called_types = make_type_vector<T>();
    return x;
  }

  template <typename T, typename U>
  void deduced_and_nondeduced(const U& /*unused*/) const {
    called_func = "B::deduced_and_nondeduced";
    called_types = make_type_vector<T, U>();
  }
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.FakeVirtual", "[Unit][Utilities]") {
  // Make sure the code example works
  {
    Derived d;
    CHECK(d.fv<int>(0) == 3);
    CHECK(static_cast<Base&>(d).fv<double>(1) == 4);
  }

  const std::unique_ptr<MultipleBase> a = std::make_unique<A>();
  const std::unique_ptr<MultipleBase> b = std::make_unique<B>();

  CHECK(a->deduced(3) == 3);
  CHECK(called_func == "A::deduced");
  CHECK(called_types == make_type_vector<int>());

  CHECK(b->deduced(7.5) == 7.5);
  CHECK(called_func == "B::deduced");
  CHECK(called_types == make_type_vector<double>());

  {
    int x;
    // Checks that references are passed in and out correctly.
    CHECK(&a->nondeduced<int>(x) == &x);
    CHECK(called_func == "A::nondeduced");
    CHECK(called_types == make_type_vector<int>());

    CHECK(&b->nondeduced<double>(x) == &x);
    CHECK(called_func == "B::nondeduced");
    CHECK(called_types == make_type_vector<double>());
  }

  a->deduced_and_nondeduced<int>(3.5);
  CHECK(called_func == "A::deduced_and_nondeduced");
  CHECK(called_types == (make_type_vector<int, double>()));

  b->deduced_and_nondeduced<int>(3);
  CHECK(called_func == "B::deduced_and_nondeduced");
  CHECK(called_types == (make_type_vector<int, int>()));
}

namespace {
class C : public MultipleBase::Inherit {
 public:
  template <typename T>
  T deduced(const T& /*unused*/) /* not const for testing */ {
    called_func = "C::deduced";
    called_types = make_type_vector<>();
  }
};
}  // namespace

// Check that you get an error when trying to call an unregistered
// class's function through the base class.
// [[OutputRegex, is not registered with]]
SPECTRE_TEST_CASE("Unit.Utilities.FakeVirtual.Unregistered",
                  "[Unit][Utilities]") {
  ERROR_TEST();
  const std::unique_ptr<MultipleBase> c = std::make_unique<C>();

  c->deduced(1);
}

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

  CHECK(11 ==
        (call_with_dynamic_type<int, tmpl::list<D1, D2>>(
             &cpp17::as_const(*d1),
             [](const auto* const p) { return p->const_func(); })));
  CHECK(12 ==
        (call_with_dynamic_type<int, tmpl::list<D1, D2>>(
             &cpp17::as_const(*d2),
             [](const auto* const p) { return p->const_func(); })));

  // Test void return type
  int result = 0;
  call_with_dynamic_type<void, tmpl::list<D1, D2>>(
      d2.get(), [&result](auto* const p) { result = p->mutable_func(); });
  CHECK(2 == result);
}
