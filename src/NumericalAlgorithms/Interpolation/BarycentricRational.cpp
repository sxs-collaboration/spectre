// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/BarycentricRational.hpp"

#include <algorithm>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>

#include "ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace intrp {
BarycentricRational::BarycentricRational(std::vector<double> x_values,
                                         std::vector<double> y_values,
                                         const size_t order) noexcept
    : x_values_(std::move(x_values)),
      y_values_(std::move(y_values)),
      weights_(x_values_.size(), 0.0),
      order_(static_cast<ssize_t>(order)) {
  ASSERT(x_values_.size() == y_values_.size(),
         "The x-value and y-value vectors must be of the same length, but "
         "receives x-value of size: "
             << x_values_.size()
             << " and y-value of size: " << y_values_.size());
  ASSERT(x_values_.size() >= order,
         " The size of the x-value and y-value vectors must be at least the "
         "requested order of interpolation. The requested order is: "
             << order_
             << " while the size of the vectors is: " << x_values_.size());
  // Use ssize_t because we do signed arithmetic
  const auto size = static_cast<ssize_t>(x_values_.size());
  // for each weights[k]...
  for (ssize_t k = 0; k < size; ++k) {
    const ssize_t i_lower_bound = std::max(k - order_, ssize_t{0});
    const ssize_t i_upper_bound = std::min(size - order_ - 1, k);
    // perform sum w_k = \sum_{i=k-order, 0 \le i < N - order}^{k} (-1)^i where
    // N is the size of the input vectors, order the interpolation order, k is
    // the index of the weights (which is in [0, order_]).
    for (ssize_t i = i_lower_bound; i <= i_upper_bound; ++i) {
      const ssize_t j_max = std::min(i + order_, size - 1);
      double inv_product = 1.0;
      // perform product: \prod_{j=i, j!=k}^{i+order} 1 / (x_k - x_j)
      for (ssize_t j = i; j <= j_max; ++j) {
        if (UNLIKELY(j == k)) {
          continue;
        }
        inv_product *= (x_values_[static_cast<size_t>(k)] -
                        x_values_[static_cast<size_t>(j)]);
      }
      if ((i & static_cast<ssize_t>(1)) == 1) {
        // if i is odd subtract
        weights_[static_cast<size_t>(k)] -= 1.0 / inv_product;
      } else {
        // if i is even add
        weights_[static_cast<size_t>(k)] += 1.0 / inv_product;
      }
    }
  }
}

double BarycentricRational::operator()(const double x_to_interp_to) const
    noexcept {
  double numerator = 0.0;
  double denominator = 0.0;
  const size_t size = x_values_.size();
  for (size_t i = 0; i < size; ++i) {
    if (x_to_interp_to == x_values_[i]) {
      return y_values_[i];
    }
    const double temp = weights_[i] / (x_to_interp_to - x_values_[i]);
    numerator += temp * y_values_[i];
    denominator += temp;
  }
  return numerator / denominator;
}

size_t BarycentricRational::order() const noexcept {
  return static_cast<size_t>(order_);
}

void BarycentricRational::pup(PUP::er& p) noexcept {
  p | x_values_;
  p | y_values_;
  p | weights_;
  p | order_;
}
}  // namespace intrp
