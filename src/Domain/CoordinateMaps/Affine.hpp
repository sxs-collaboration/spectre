// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class Affine.

#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain {
namespace CoordinateMaps {

/*!
 * \ingroup CoordinateMapsGroup
 * \brief Affine map from \f$\xi \in [A, B]\rightarrow x \in [a, b]\f$.
 *
 * The formula for the mapping is...
 * \f[
 * x = \frac{b}{B-A} (\xi-A) +\frac{a}{B-A}(B-\xi)
 * \f]
 * \f[
 * \xi =\frac{B}{b-a} (x-a) +\frac{A}{b-a}(b-x)
 * \f]
 */
class Affine {
 public:
  static constexpr size_t dim = 1;

  Affine(double A, double B, double a, double b);

  Affine() = default;
  ~Affine() = default;
  Affine(const Affine&) = default;
  Affine(Affine&&) = default;  // NOLINT
  Affine& operator=(const Affine&) = default;
  Affine& operator=(Affine&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 1> operator()(
      const std::array<T, 1>& source_coords) const;

  /// The inverse function is only callable with doubles because the inverse
  /// might fail if called for a point out of range, and it is unclear
  /// what should happen if the inverse were to succeed for some points in a
  /// DataVector but fail for other points.
  std::optional<std::array<double, 1>> inverse(
      const std::array<double, 1>& target_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> jacobian(
      const std::array<T, 1>& source_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> inv_jacobian(
      const std::array<T, 1>& source_coords) const;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p);  // NOLINT

  bool is_identity() const { return is_identity_; }

 private:
  friend bool operator==(const Affine& lhs, const Affine& rhs);

  double A_{-1.0};
  double B_{1.0};
  double a_{-1.0};
  double b_{1.0};
  double length_of_domain_{2.0};  // B-A
  double length_of_range_{2.0};   // b-a
  double jacobian_{length_of_range_ / length_of_domain_};
  double inverse_jacobian_{length_of_domain_ / length_of_range_};
  bool is_identity_{false};
};

inline bool operator!=(const CoordinateMaps::Affine& lhs,
                       const CoordinateMaps::Affine& rhs) {
  return not(lhs == rhs);
}

}  // namespace CoordinateMaps
}  // namespace domain
