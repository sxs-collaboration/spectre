// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <functional>
#include <type_traits>

#include "Utilities/DereferenceWrapper.hpp"

static_assert(
    std::is_same<decltype(dereference_wrapper(std::declval<double>())),
                 double&&>::value,
    "Failed testing dereference_wrapper");
static_assert(
    std::is_same<decltype(dereference_wrapper(std::declval<const double>())),
                 const double&&>::value,
    "Failed testing dereference_wrapper");
static_assert(
    std::is_same<decltype(dereference_wrapper(std::declval<double&>())),
                 double&>::value,
    "Failed testing dereference_wrapper");
static_assert(
    std::is_same<decltype(dereference_wrapper(std::declval<const double&>())),
                 const double&>::value,
    "Failed testing dereference_wrapper");

static_assert(std::is_same<decltype(dereference_wrapper(
                               std::declval<std::reference_wrapper<double>>())),
                           double&&>::value,
              "Failed testing dereference_wrapper");
static_assert(
    std::is_same<decltype(dereference_wrapper(
                     std::declval<const std::reference_wrapper<double>>())),
                 double&&>::value,
    "Failed testing dereference_wrapper");
static_assert(
    std::is_same<decltype(dereference_wrapper(
                     std::declval<std::reference_wrapper<double>&>())),
                 double&>::value,
    "Failed testing dereference_wrapper");
static_assert(
    std::is_same<decltype(dereference_wrapper(
                     std::declval<const std::reference_wrapper<double>&>())),
                 double&>::value,
    "Failed testing dereference_wrapper");

SPECTRE_TEST_CASE("Unit.Utilities.DereferenceWrapper", "[Unit][Utilities]") {
  double a = 1.478;
  std::reference_wrapper<double> ref_a(a);
  CHECK(dereference_wrapper(ref_a) == 1.478);
  std::reference_wrapper<const double> ref_ca(a);
  CHECK(dereference_wrapper(ref_ca) == 1.478);
  const std::reference_wrapper<double> cref_a(a);
  CHECK(dereference_wrapper(cref_a) == 1.478);
  const std::reference_wrapper<const double> cref_ca(a);
  CHECK(dereference_wrapper(cref_ca) == 1.478);

  CHECK(dereference_wrapper(a) == 1.478);
  CHECK(dereference_wrapper(static_cast<const double&>(a)) == 1.478);
  CHECK(dereference_wrapper(static_cast<double&>(a)) == 1.478);
  CHECK(dereference_wrapper(static_cast<const double&&>(a)) == 1.478);
  CHECK(dereference_wrapper(static_cast<double&&>(a)) == 1.478);
}
