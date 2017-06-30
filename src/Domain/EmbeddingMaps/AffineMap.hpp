
/// \file
/// Defines the class AffineMap.

#pragma once

#include <memory>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/EmbeddingMaps/EmbeddingMap.hpp"
#include "Parallel/CharmPupable.hpp"

namespace EmbeddingMaps {

/// Linear map from \f$\xi \in [A, B]\rightarrow x \in [a, b]\f$.
/// The formula for the mapping is...
/*! \f[
  x = \frac{b}{B-A} (\xi-A) +\frac{a}{B-A}(B-\xi)
 \f]
 \f[
  \xi =\frac{B}{b-a} (x-a) +\frac{A}{b-a}(b-x)
 \f]
*/
class AffineMap : public EmbeddingMap<1, 1> {
 public:
  AffineMap(const double A, const double B, const double a,
                       const double b)
      : A_(A), B_(B), a_(a), b_(b), L_(B - A), l_(b - a) {}

  AffineMap() = default;
  ~AffineMap() override = default;
  AffineMap(const AffineMap&) = delete;
  AffineMap(AffineMap&&) noexcept = default;  // NOLINT
  AffineMap& operator=(const AffineMap&) = delete;
  AffineMap& operator=(AffineMap&&) = default;

  std::unique_ptr<EmbeddingMap<1, 1>> get_clone() const override {
    return std::make_unique<AffineMap>(A_, B_, a_, b_);
  }

  Point<1, Frame::Grid> operator()(const Point<1, Frame::Logical>& xi) const {
    return Point<1, Frame::Grid>((l_ * xi[0] + a_ * B_ - b_ * A_) / L_);
  }

  Point<1, Frame::Logical> inverse(const Point<1, Frame::Grid>& x) const {
    return Point<1, Frame::Logical>((L_ * x[0] - a_ * B_ + b_ * A_) / l_);
  }

  double jacobian(const Point<1, Frame::Logical>& /* xi */,
                             size_t ud, size_t ld) const {
    ASSERT(0 == ld, "ld = " << ld);
    ASSERT(0 == ud, "ud = " << ud);
    return l_ / L_;
  }

  double inv_jacobian(const Point<1, Frame::Logical>& /*xi*/, size_t ud,
                      size_t ld) const override {
    ASSERT(0 == ld, "ld = " << ld);
    ASSERT(0 == ud, "ud = " << ud);
    return L_ / l_;
  }

  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(EmbeddingMap<1, 1>),  // NOLINT
                                     AffineMap);

  explicit AffineMap(CkMigrateMessage* /* m */)
      : A_(-1), B_(1), a_(-1), b_(1), L_(B_ - A_), l_(b_ - a_) {}

  void pup(PUP::er& p) override {
    EmbeddingMap<1, 1>::pup(p);
    p | A_;
    p | B_;
    p | a_;
    p | b_;
    p | L_;
    p | l_;
  }

 private:
  double A_{-1.0};
  double B_{1.0};
  double a_{-1.0};
  double b_{1.0};
  double L_{2.0};  // B-A
  double l_{2.0};  // b-a
};

bool operator==(const AffineMap& lhs, const AffineMap& rhs);
bool operator!=(const AffineMap& lhs, const AffineMap& rhs);
}  // namespace EmbeddingMaps

/// \cond HIDDEN_SYMBOLS
PUP::able::PUP_ID EmbeddingMaps::AffineMap::my_PUP_ID = 0;  // NOLINT
/// \endcond
