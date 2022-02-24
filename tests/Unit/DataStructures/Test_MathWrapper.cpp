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
}
