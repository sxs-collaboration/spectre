// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <sharp_cxx.h>

#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Spectral {
namespace Swsh {
namespace {

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Spectral.SwshCoefficients",
                  "[Unit][NumericalAlgorithms]") {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> sdist{8, 64};
  const size_t l_max = sdist(gen);

  CAPTURE(l_max);
  const detail::Coefficients& precomputed_coefficients =
      detail::precomputed_coefficients(l_max);

  const detail::Coefficients& another_precomputed_coefficients =
      detail::precomputed_coefficients(l_max);

  // checks that the same pointer is in both
  CHECK(precomputed_coefficients.get_sharp_alm_info() ==
        another_precomputed_coefficients.get_sharp_alm_info());

  const detail::Coefficients computed_coefficients{l_max};

  CHECK(precomputed_coefficients.l_max() == l_max);
  CHECK(computed_coefficients.l_max() == l_max);

  sharp_alm_info* manual_sai;
  sharp_make_triangular_alm_info(l_max, l_max, 1, &manual_sai);

  // check that all of the precomputed coefficients, the directly constructed
  // coefficients, and the manually created sharp_alm_info* all contain the same
  // data
  CHECK(precomputed_coefficients.get_sharp_alm_info()->lmax ==
        computed_coefficients.get_sharp_alm_info()->lmax);
  CHECK(precomputed_coefficients.get_sharp_alm_info()->lmax ==
        manual_sai->lmax);
  CHECK(precomputed_coefficients.get_sharp_alm_info()->lmax ==
        computed_coefficients.l_max());

  CHECK(precomputed_coefficients.get_sharp_alm_info()->nm ==
        computed_coefficients.get_sharp_alm_info()->nm);
  CHECK(precomputed_coefficients.get_sharp_alm_info()->nm == manual_sai->nm);

  CHECK(precomputed_coefficients.get_sharp_alm_info()->flags ==
        computed_coefficients.get_sharp_alm_info()->flags);
  CHECK(precomputed_coefficients.get_sharp_alm_info()->flags ==
        manual_sai->flags);

  CHECK(precomputed_coefficients.get_sharp_alm_info()->stride ==
        computed_coefficients.get_sharp_alm_info()->stride);
  CHECK(precomputed_coefficients.get_sharp_alm_info()->stride ==
        manual_sai->stride);

  for (size_t m_index = 0;
       m_index <
       static_cast<size_t>(precomputed_coefficients.get_sharp_alm_info()->nm);
       ++m_index) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(precomputed_coefficients.get_sharp_alm_info()->mval[m_index] ==
          computed_coefficients.get_sharp_alm_info()->mval[m_index]);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(precomputed_coefficients.get_sharp_alm_info()->mval[m_index] ==
          manual_sai->mval[m_index]);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(precomputed_coefficients.get_sharp_alm_info()->mvstart[m_index] ==
          computed_coefficients.get_sharp_alm_info()->mvstart[m_index]);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(precomputed_coefficients.get_sharp_alm_info()->mvstart[m_index] ==
          manual_sai->mvstart[m_index]);
  }
}

// [[OutputRegex, is not below the maximum l_max]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Spectral.SwshCoefficients.PrecomputationOverrun",
    "[Unit][NumericalAlgorithms]") {
  ERROR_TEST();
  detail::precomputed_coefficients(detail::coefficients_maximum_l_max + 1);
  ERROR("Failed to trigger ERROR in an error test");
}
}  // namespace
}  // namespace Swsh
}  // namespace Spectral
