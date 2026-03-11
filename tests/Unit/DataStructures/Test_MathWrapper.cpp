// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/MathWrapper.hpp"
#include "Helpers/DataStructures/MathWrapper.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename T>
void test_into_math_wrapper_type_scalar(T value) {
  const auto copy = value;
  auto type_erased = into_math_wrapper_type(std::move(value));
  static_assert(std::is_same_v<decltype(type_erased), math_wrapper_type<T>>);
  CHECK(type_erased == copy);
}

template <typename T>
void test_into_math_wrapper_type_vector(T value) {
  const auto* const data = value.data();
  auto type_erased = into_math_wrapper_type(std::move(value));
  static_assert(std::is_same_v<decltype(type_erased), math_wrapper_type<T>>);
  CHECK(type_erased.data() == data);
}

SPECTRE_TEST_CASE("Unit.Utilities.MathWrapper", "[Unit][Utilities]") {
  TestHelpers::MathWrapper::test_type<double>(1.0, 2.0, 3.0);
  TestHelpers::MathWrapper::test_type<std::complex<double>>(
      {1.0, 2.0}, {3.0, 4.0}, std::complex<double>{5.0, 6.0});
  TestHelpers::MathWrapper::test_type<DataVector>({1.0, 2.0}, {3.0, 4.0}, 5.0);
  TestHelpers::MathWrapper::test_type<ComplexDataVector>(
      {std::complex<double>{1.0, 2.0}, std::complex<double>{3.0, 4.0}},
      {std::complex<double>{5.0, 6.0}, std::complex<double>{7.0, 8.0}},
      std::complex<double>{9.0, 10.0});
  TestHelpers::MathWrapper::test_type<std::array<double, 2>>({1.0, 2.0},
                                                             {3.0, 4.0}, 5.0);

  // [MathWrapper]
  double mutable_double = 1.0;
  const double const_double = 2.0;
  const auto mutable_wrapper = make_math_wrapper(&mutable_double);
  const auto const_wrapper = make_math_wrapper(const_double);
  *mutable_wrapper += *const_wrapper;
  CHECK(mutable_double == 3.0);
  // [MathWrapper]

  test_into_math_wrapper_type_scalar(3.4);
  test_into_math_wrapper_type_scalar(std::complex{3.4, 5.6});
  test_into_math_wrapper_type_vector(DataVector{1.2, 3.4, 5.6});
  test_into_math_wrapper_type_vector(
      ComplexDataVector{std::complex{1.2, 3.4}, std::complex{5.6, 7.8}});
}
}  // namespace
