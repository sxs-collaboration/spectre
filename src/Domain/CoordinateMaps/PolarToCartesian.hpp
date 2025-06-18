// Distributed under the MIT License.
// See LICENSE.txt for details.

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

namespace domain::CoordinateMaps {

/*!
 * \ingroup CoordinateMapsGroup
 *
 * \brief Transformation from polar to Cartesian coordinates.
 *
 * \details This is a mapping from \f$(r,\phi) \rightarrow (x,y) \f$.
 *
 * The formula for the mapping is...
 * \f{eqnarray*}
 *     x &=& r \cos\phi \\
 *     y &=& r \sin\phi
 * \f}
 */
class PolarToCartesian {
 public:
  static constexpr size_t dim = 2;
  PolarToCartesian();
  ~PolarToCartesian() = default;
  PolarToCartesian(PolarToCartesian&&);
  PolarToCartesian(const PolarToCartesian&);
  PolarToCartesian& operator=(const PolarToCartesian&);
  PolarToCartesian& operator=(PolarToCartesian&&);

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 2> operator()(
      const std::array<T, 2>& source_coords) const;

  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  std::optional<std::array<double, 2>> inverse(
      const std::array<double, 2>& target_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame> jacobian(
      const std::array<T, 2>& source_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame> inv_jacobian(
      const std::array<T, 2>& source_coords) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  static constexpr bool is_identity() { return false; }
};

bool operator==(const PolarToCartesian& lhs, const PolarToCartesian& rhs);

bool operator!=(const PolarToCartesian& lhs, const PolarToCartesian& rhs);
}  // namespace domain::CoordinateMaps
