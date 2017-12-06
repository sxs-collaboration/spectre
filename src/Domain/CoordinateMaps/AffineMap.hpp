// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class AffineMap.

#pragma once

#include <array>
#include <memory>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/TypeTraits.hpp"

namespace CoordinateMaps {

/*!
 * \ingroup CoordinateMapsGroup
 * \brief Linear map from \f$\xi \in [A, B]\rightarrow x \in [a, b]\f$.
 *
 * The formula for the mapping is...
 * \f[
 * x = \frac{b}{B-A} (\xi-A) +\frac{a}{B-A}(B-\xi)
 * \f]
 * \f[
 * \xi =\frac{B}{b-a} (x-a) +\frac{A}{b-a}(b-x)
 * \f]
*/
class AffineMap {
 public:
  static constexpr size_t dim = 1;

  AffineMap(double A, double B, double a, double b);

  AffineMap() = default;
  ~AffineMap() = default;
  AffineMap(const AffineMap&) = default;
  AffineMap(AffineMap&&) noexcept = default;  // NOLINT
  AffineMap& operator=(const AffineMap&) = default;
  AffineMap& operator=(AffineMap&&) = default;

  template <typename T>
  std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 1> operator()(
      const std::array<T, 1>& xi) const;

  template <typename T>
  std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 1> inverse(
      const std::array<T, 1>& x) const;

  template <typename T>
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
  inv_jacobian(const std::array<T, 1>& /*xi*/) const;

  template <typename T>
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
  jacobian(const std::array<T, 1>& /*xi*/) const;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p);  // NOLINT

 private:
  friend bool operator==(const AffineMap& lhs, const AffineMap& rhs) noexcept;

  double A_{-1.0};
  double B_{1.0};
  double a_{-1.0};
  double b_{1.0};
  double length_of_domain_{2.0};  // B-A
  double length_of_range_{2.0};   // b-a
  double jacobian_{length_of_range_ / length_of_domain_};
  double inverse_jacobian_{length_of_domain_ / length_of_range_};
};

inline bool operator!=(const CoordinateMaps::AffineMap& lhs,
                       const CoordinateMaps::AffineMap& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace CoordinateMaps
