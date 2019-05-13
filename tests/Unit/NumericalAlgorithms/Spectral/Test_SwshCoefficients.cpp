// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <complex>
#include <cstddef>
#include <limits>
#include <random>
#include <sharp_cxx.h>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/Spectral/ComplexDataView.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/NumericalAlgorithms/Spectral/SwshTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Spectral {
namespace Swsh {
namespace {

void test_swsh_coefficients_class_interface() noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> sdist{8, 64};
  const size_t l_max = sdist(gen);

  CAPTURE(l_max);
  const CoefficientsMetadata& precomputed_libsharp_lm =
      cached_coefficients_metadata(l_max);

  const CoefficientsMetadata& another_precomputed_libsharp_lm =
      cached_coefficients_metadata(l_max);

  // checks that the same pointer is in both
  CHECK(precomputed_libsharp_lm.get_sharp_alm_info() ==
        another_precomputed_libsharp_lm.get_sharp_alm_info());

  const CoefficientsMetadata computed_coefficients{l_max};

  CHECK(precomputed_libsharp_lm.l_max() == l_max);
  CHECK(computed_coefficients.l_max() == l_max);

  sharp_alm_info* expected_sharp_alm_info;
  sharp_make_triangular_alm_info(l_max, l_max, 1, &expected_sharp_alm_info);

  // check that all of the precomputed coefficients, the directly constructed
  // coefficients, and the manually created sharp_alm_info* all contain the same
  // data
  CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->lmax ==
        computed_coefficients.get_sharp_alm_info()->lmax);
  CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->lmax ==
        expected_sharp_alm_info->lmax);
  CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->lmax ==
        computed_coefficients.l_max());

  CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->nm ==
        computed_coefficients.get_sharp_alm_info()->nm);
  CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->nm ==
        expected_sharp_alm_info->nm);

  CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->flags ==
        computed_coefficients.get_sharp_alm_info()->flags);
  CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->flags ==
        expected_sharp_alm_info->flags);

  CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->stride ==
        computed_coefficients.get_sharp_alm_info()->stride);
  CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->stride ==
        expected_sharp_alm_info->stride);

  for (size_t m_index = 0;
       m_index <
       static_cast<size_t>(precomputed_libsharp_lm.get_sharp_alm_info()->nm);
       ++m_index) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->mval[m_index] ==
          computed_coefficients.get_sharp_alm_info()->mval[m_index]);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->mval[m_index] ==
          expected_sharp_alm_info->mval[m_index]);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->mvstart[m_index] ==
          computed_coefficients.get_sharp_alm_info()->mvstart[m_index]);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->mvstart[m_index] ==
          expected_sharp_alm_info->mvstart[m_index]);
  }

  CHECK(precomputed_libsharp_lm.begin() == precomputed_libsharp_lm.cbegin());
  CHECK(precomputed_libsharp_lm.end() == precomputed_libsharp_lm.cend());
  CHECK(precomputed_libsharp_lm.begin() != precomputed_libsharp_lm.end());

  size_t offset_counter = 0;
  size_t expected_l = 0;
  size_t expected_m = 0;
  for (const auto& coefficient_info : precomputed_libsharp_lm) {
    CHECK(coefficient_info.transform_of_real_part_offset == offset_counter);
    CHECK(coefficient_info.transform_of_imag_part_offset -
              coefficient_info.transform_of_real_part_offset ==
          size_of_libsharp_coefficient_vector(l_max) / 2);
    CHECK(coefficient_info.l == expected_l);
    CHECK(coefficient_info.m == expected_m);
    ++offset_counter;
    if (expected_l == l_max) {
      ++expected_m;
      expected_l = expected_m;
    } else {
      ++expected_l;
    }
  }
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Spectral.SwshCoefficients",
                  "[Unit][NumericalAlgorithms]") {
  {
    INFO("Checking Coefficients for libsharp interoperability");
    test_swsh_coefficients_class_interface();
  }
}

// [[OutputRegex, is not below the maximum l_max]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Spectral.SwshCoefficients.PrecomputationOverrun",
    "[Unit][NumericalAlgorithms]") {
  ERROR_TEST();
  cached_coefficients_metadata(detail::coefficients_maximum_l_max + 1);
  ERROR("Failed to trigger ERROR in an error test");
}
}  // namespace
}  // namespace Swsh
}  // namespace Spectral
