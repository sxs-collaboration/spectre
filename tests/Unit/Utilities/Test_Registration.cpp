// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <memory>

#include "Options/Options.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestCreation.hpp"

namespace {
/// [registrar_structure]
template <typename Registrars>
class Base {
 public:
  Base() = default;
  Base(const Base&) = default;
  Base(Base&&) = default;
  Base& operator=(const Base&) = default;
  Base& operator=(Base&&) = default;
  virtual ~Base() = default;
  using creatable_classes = Registration::registrants<Registrars>;

  virtual int func() const noexcept = 0;
};

template <typename Registrars>
class Derived1;

namespace Registrars {
using Derived1 = Registration::Registrar<::Derived1>;
}  // namespace Registrars

template <typename Registrars = tmpl::list<Registrars::Derived1>>
class Derived1 : public Base<Registrars> {
 public:
  static constexpr OptionString help = "help";
  using options = tmpl::list<>;
  int func() const noexcept override { return 1; }
};
/// [registrar_structure]

/// [registrar]
template <typename SomeArg, typename Registrars>
class Derived2;

namespace Registrars {
template <typename SomeArg>
using Derived2 = Registration::Registrar<::Derived2, SomeArg>;
}  // namespace Registrars

template <typename SomeArg,
          typename Registrars = tmpl::list<Registrars::Derived2<SomeArg>>>
class Derived2 : public Base<Registrars> {
 public:
  static constexpr OptionString help = "help";
  using options = tmpl::list<>;
  int func() const noexcept override { return 2; }
};
/// [registrar]

/// [custom_registrar]
template <int N, typename Registrars>
class Derived3;

namespace Registrars {
template <int N>
struct Derived3 {
  template <typename RegistrarList>
  using f = ::Derived3<N, RegistrarList>;
};
}  // namespace Registrars

template <int N, typename Registrars = tmpl::list<Registrars::Derived3<N>>>
class Derived3 : public Base<Registrars> {
 public:
  static constexpr OptionString help = "help";
  using options = tmpl::list<>;
  int func() const noexcept override { return N; }
};
/// [custom_registrar]
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.Registration", "[Unit][Utilities]") {
  /// [registrar_use]
  using ConcreteBase =
      Base<tmpl::list<Registrars::Derived1, Registrars::Derived2<double>,
                      Registrars::Derived3<4>>>;
  /// [registrar_use]
  CHECK(test_factory_creation<ConcreteBase>("  Derived1")->func() == 1);
  CHECK(test_factory_creation<ConcreteBase>("  Derived2")->func() == 2);
  CHECK(test_factory_creation<ConcreteBase>("  Derived3")->func() == 4);

  // Test standalone derived classes
  CHECK(Derived1<>{}.func() == 1);
  CHECK(Derived2<double>{}.func() == 2);
  CHECK(Derived3<4>{}.func() == 4);
}
