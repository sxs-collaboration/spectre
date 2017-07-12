// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class AffineMap.

#pragma once

#include <memory>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/EmbeddingMaps/EmbeddingMap.hpp"
#include "Parallel/CharmPupable.hpp"

namespace EmbeddingMaps {

/*! \ingroup EmbeddingMaps
 * Linear map from \f$\xi \in [A, B]\rightarrow x \in [a, b]\f$.
 * The formula for the mapping is...
 * \f[
 * x = \frac{b}{B-A} (\xi-A) +\frac{a}{B-A}(B-\xi)
 * \f]
 * \f[
 * \xi =\frac{B}{b-a} (x-a) +\frac{A}{b-a}(b-x)
 * \f]
*/
class AffineMap : public EmbeddingMap<1, 1> {
 public:
  AffineMap(double A, double B, double a, double b);

  AffineMap() = default;
  ~AffineMap() override = default;
  AffineMap(const AffineMap&) = delete;
  AffineMap(AffineMap&&) noexcept = default;  // NOLINT
  AffineMap& operator=(const AffineMap&) = delete;
  AffineMap& operator=(AffineMap&&) = default;

  std::unique_ptr<EmbeddingMap<1, 1>> get_clone() const override;

  Point<1, Frame::Grid> operator()(
      const Point<1, Frame::Logical>& xi) const override;

  Point<1, Frame::Logical> inverse(
      const Point<1, Frame::Grid>& x) const override;

  double jacobian(const Point<1, Frame::Logical>& xi, size_t ud,
                  size_t ld) const override;

  double inv_jacobian(const Point<1, Frame::Logical>& xi, size_t ud,
                      size_t ld) const override;

  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(EmbeddingMap<1, 1>),  // NOLINT
                                     AffineMap);

  explicit AffineMap(CkMigrateMessage* /* m */);

  void pup(PUP::er& p) override;

 private:
  double A_{-1.0};
  double B_{1.0};
  double a_{-1.0};
  double b_{1.0};
  double length_of_domain_{2.0};  // B-A
  double length_of_range_{2.0};   // b-a
  double jacobian_;
  double inverse_jacobian_;
};
}  // namespace EmbeddingMaps
