// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/EmbeddingMaps/AffineMap.hpp"

namespace EmbeddingMaps {

AffineMap::AffineMap(const double A, const double B, const double a,
                     const double b)
    : A_(A),
      B_(B),
      a_(a),
      b_(b),
      length_of_domain_(B - A),
      length_of_range_(b - a),
      jacobian_(length_of_range_ / length_of_domain_),
      inverse_jacobian_(length_of_domain_ / length_of_range_) {}

std::unique_ptr<EmbeddingMap<1, 1>> AffineMap::get_clone() const {
  return std::make_unique<AffineMap>(A_, B_, a_, b_);
}

Point<1, Frame::Grid> AffineMap::operator()(
    const Point<1, Frame::Logical>& xi) const {
  return Point<1, Frame::Grid>((length_of_range_ * xi[0] + a_ * B_ - b_ * A_) /
                               length_of_domain_);
}

Point<1, Frame::Logical> AffineMap::inverse(
    const Point<1, Frame::Grid>& x) const {
  return Point<1, Frame::Logical>(
      (length_of_domain_ * x[0] - a_ * B_ + b_ * A_) / length_of_range_);
}
double AffineMap::jacobian(const Point<1, Frame::Logical>& /* xi */, size_t ud,
                           size_t ld) const {
  ASSERT(0 == ld, "ld = " << ld);
  ASSERT(0 == ud, "ud = " << ud);
  return jacobian_;
}

double AffineMap::inv_jacobian(const Point<1, Frame::Logical>& /* xi */,
                               size_t ud, size_t ld) const {
  ASSERT(0 == ld, "ld = " << ld);
  ASSERT(0 == ud, "ud = " << ud);
  return inverse_jacobian_;
}

AffineMap::AffineMap(CkMigrateMessage* /* m */)
    : A_(-1),
      B_(1),
      a_(-1),
      b_(1),
      length_of_domain_(B_ - A_),
      length_of_range_(b_ - a_),
      jacobian_(length_of_range_ / length_of_domain_),
      inverse_jacobian_(length_of_domain_ / length_of_range_) {}

void AffineMap::pup(PUP::er& p) {
  EmbeddingMap<1, 1>::pup(p);
  p | A_;
  p | B_;
  p | a_;
  p | b_;
  p | length_of_domain_;
  p | length_of_range_;
  p | jacobian_;
  p | inverse_jacobian_;
}
}  // namespace EmbeddingMaps

/// \cond HIDDEN_SYMBOLS
PUP::able::PUP_ID EmbeddingMaps::AffineMap::my_PUP_ID = 0;  // NOLINT
/// \endcond
