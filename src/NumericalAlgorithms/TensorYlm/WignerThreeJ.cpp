// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/TensorYlm/WignerThreeJ.hpp"

#include <limits>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/ExpectsAndEnsures.hpp"

extern "C" {
void drc3jj_(const double& l2, const double& l3, const double& m2,
             const double& m3, double& l1min, double& l1max, double* answer,
             const int& answersize, int& error_code, const double& doublemax);
}

WignerThreeJ::WignerThreeJ(const size_t l2, const int m2, const size_t l3,
                           const int m3)
    : l2_(static_cast<double>(l2)),
      m2_(m2),
      l3_(static_cast<double>(l3)),
      m3_(m3),
      l1_min_(static_cast<size_t>(
          std::max(std::abs(static_cast<int>(l2) - static_cast<int>(l3)),
                   std::abs(m2 + m3)))),
      l1_max_(l2 + l3),
      up_to_date_(false),
      coefs_(l1_max_ - l1_min_ + 1) {
  if (UNLIKELY(std::abs(m2) > static_cast<int>(l2))) {
    ERROR("WignerThreeJ: Must have |m2| <= l2. m2 is " << m2 << " but l2 is "
                                                       << l2);
  }
  if (UNLIKELY(std::abs(m3) > static_cast<int>(l3))) {
    ERROR("WignerThreeJ: Must have |m3| <= l3. m3 is " << m3 << " but l3 is "
                                                       << l3);
  }
}

double WignerThreeJ::operator()(const size_t l1) {
  if (l1 < l1_min_ or l1 > l1_max_) {
    return 0.0;
  }
  if (not up_to_date_) {
    recompute();
  }
  return coefs_[l1 - l1_min_];
}

void WignerThreeJ::recompute() {
  // error_code, l1min, and l1max are output parameters of drc3jj.
  // We don't care about l1min and l1max.
  int error_code = 0;
  double l1min = 0.0;
  double l1max = 0.0;

  drc3jj_(l2_, l3_, m2_, m3_, l1min, l1max, coefs_.data(),
          static_cast<int>(coefs_.size()), error_code,
          std::numeric_limits<double>::max());

  if (UNLIKELY(error_code != 0)) {
    ERROR("Nonzero error code "
          << error_code
          << ", codes are:\n"
             "IER=0 No errors.\n"
             "IER=1 Either L2.LT.ABS(M2) or L3.LT.ABS(M3).\n"
             "IER=2 Either L2+ABS(M2) or L3+ABS(M3) non-integer.\n"
             "IER=3 L1MAX-L1MIN not an integer.\n"
             "IER=4 L1MAX less than L1MIN.\n"
             "IER=5 NDIM less than L1MAX-L1MIN+1.\n");
  }
  up_to_date_ = true;
}
