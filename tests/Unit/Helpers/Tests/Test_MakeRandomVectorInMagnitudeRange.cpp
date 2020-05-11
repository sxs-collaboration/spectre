// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeRandomVectorInMagnitudeRange.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"

void check_random_values(const double v, const double bound1,
                         const double bound2) noexcept {
  CHECK(approx(v) >= std::min(bound1, bound2));
  CHECK(approx(v) <= std::max(bound1, bound2));
}

// clang-tidy is wrong, this is a function definition
template <typename T>
void check_random_values(const T& c,                      // NOLINT
                         const double bound1,             // NOLINT
                         const double bound2) noexcept {  // NOLINT
  for (const auto& v : c) {
    check_random_values(v, bound1, bound2);
  }
}

template <typename... Tags>
void check_random_values(const Variables<tmpl::list<Tags...>>& v,
                         const double bound1, const double bound2) noexcept {
  expand_pack((
      check_random_values<decltype(get<Tags>(v))>(get<Tags>(v), bound1, bound2),
      cpp17::void_type{})...);
}

// Compute Magnitude
template <typename DataType, size_t Dim, typename Fr>
Scalar<DataType> magnitude_auto_invert(
    const tnsr::I<DataType, Dim, Fr>& vector,
    const tnsr::ii<DataType, Dim, Fr>& metric) {
  return magnitude(vector, metric);
}

template <typename DataType, size_t Dim, typename Fr>
Scalar<DataType> magnitude_auto_invert(
    const tnsr::i<DataType, Dim, Fr>& covector,
    const tnsr::ii<DataType, Dim, Fr>& metric) {
  const tnsr::II<DataType, Dim, Fr> inverse_metric =
      determinant_and_inverse(metric).second;

  return magnitude(covector, inverse_metric);
}

template <size_t Dim, UpLo Ul, typename DataType>
void test_range(const gsl::not_null<std::mt19937*> generator,
                const DataType& used_for_size) noexcept {
  using Fr = Frame::Inertial;

  std::uniform_real_distribution<> distribution1(0.0, 100.0);
  double r1 = distribution1(*generator);
  std::uniform_real_distribution<> distribution2(0.0, 100.0);
  double r2 = distribution2(*generator);
  INFO("Interval is [" << r1 << "," << r2 << "]");

  //****** FIRST TEST WITH EUCLIDIAN METRIC ******//

  auto vector =
      make_random_vector_in_magnitude_range_flat<DataType, Dim, Ul, Fr>(
          generator, used_for_size, r1, r2);
  auto vector_mag = magnitude(vector);

  check_random_values(vector_mag, r1, r2);

  //****** NOW TEST WITH RANDOM METRIC ******//

  const tnsr::ii<DataType, Dim, Fr> metric =
      TestHelpers::gr::random_spatial_metric<Dim, DataType, Fr>(generator,
                                                                used_for_size);

  vector = make_random_vector_in_magnitude_range<DataType, Dim, Ul, Fr>(
      generator, metric, r1, r2);

  vector_mag = magnitude_auto_invert(vector, metric);

  check_random_values(vector_mag, r1, r2);

  //****** TEST EDGE CASES ******//

  r2 = r1;
  INFO("Interval is [" << r1 << "," << r2 << "]");

  vector = make_random_vector_in_magnitude_range<DataType, Dim, Ul, Fr>(
      generator, metric, r1, r2);

  vector_mag = magnitude_auto_invert(vector, metric);

  check_random_values(vector_mag, r1, r2);

  r1 = 0.0;
  r2 = 0.0;
  INFO("Interval is [" << r1 << "," << r2 << "]");

  vector = make_random_vector_in_magnitude_range<DataType, Dim, Ul, Fr>(
      generator, metric, r1, r2);

  vector_mag = magnitude_auto_invert(vector, metric);

  check_random_values(vector_mag, r1, r2);
}

SPECTRE_TEST_CASE("Test.TestHelpers.MakeRandomVectorInMagnitudeRange",
                  "[Unit]") {
  MAKE_GENERATOR(generator);

  SECTION("double") {
    const double d = std::numeric_limits<double>::signaling_NaN();
    test_range<1, UpLo::Up>(&generator, d);
    test_range<2, UpLo::Up>(&generator, d);
    test_range<3, UpLo::Up>(&generator, d);

    test_range<1, UpLo::Lo>(&generator, d);
    test_range<2, UpLo::Lo>(&generator, d);
    test_range<3, UpLo::Lo>(&generator, d);
  }

  SECTION("DataVector") {
    const DataVector dv(5);
    test_range<1, UpLo::Up>(&generator, dv);
    test_range<2, UpLo::Up>(&generator, dv);
    test_range<3, UpLo::Up>(&generator, dv);

    test_range<1, UpLo::Lo>(&generator, dv);
    test_range<2, UpLo::Lo>(&generator, dv);
    test_range<3, UpLo::Lo>(&generator, dv);
  }
}
