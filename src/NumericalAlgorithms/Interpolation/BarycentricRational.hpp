// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unistd.h>  // IWYU pragma: keep
#include <vector>

namespace PUP {
class er;
}  // namespace PUP

namespace intrp {
/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief A barycentric rational interpolation class
 *
 * The class builds a barycentric rational interpolant of a specified order
 * using the `x_values` and `y_values` passed into the constructor.
 * Barycentric interpolation requires \f$3N\f$ storage, and costs
 * \f$\mathcal{O}(Nd)\f$ to construct, where \f$N\f$ is the size of the x- and
 * y-value vectors and \f$d\f$ is the order of the interpolant. The evaluation
 * cost is \f$\mathcal{O}(N)\f$ compared to \f$\mathcal{O}(d)\f$ of a spline
 * method, but constructing the barycentric interpolant does not require any
 * derivatives of the function to be known.
 *
 * The interpolation function is
 *
 * \f[
 *   \mathcal{I}(x)=\frac{\sum_{i=0}^{N-1}w_i y_i /
 *   (x-x_i)}{\sum_{i=0}^{N-1}w_i/(x-x_i)}
 * \f]
 *
 * where \f$w_i\f$ are the weights. The weights are computed using
 *
 * \f[
 *   w_k=\sum_{i=k-d\\0\le i < N-d}^{k}(-1)^{i}
 *       \prod_{j=i\\j\ne k}^{i+d}\frac{1}{x_k-x_j} \f]
 *
 * \requires `x_values.size() == y_values.size()` and
 * `x_values_.size() >= order`
 */
class BarycentricRational {
 public:
  BarycentricRational() noexcept = default;
  BarycentricRational(std::vector<double> x_values,
                      std::vector<double> y_values, size_t order) noexcept;

  double operator()(double x_to_interp_to) const noexcept;

  size_t order() const noexcept;

  // clang-tidy: no runtime references
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  std::vector<double> x_values_;
  std::vector<double> y_values_;
  std::vector<double> weights_;
  ssize_t order_{0};
};
}  // namespace intrp
