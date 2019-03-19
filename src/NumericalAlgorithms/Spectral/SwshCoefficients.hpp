// Distributed under the MIT License.
// See LICENSE.txt for details

#pragma once

#include <cstdlib>
#include <memory>
#include <sharp_cxx.h>

#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "Utilities/ForceInline.hpp"

namespace Spectral {
namespace Swsh {

/// \ingroup SwshGroup
/// \brief Convenience function for determining the number of spin-weighted
/// spherical harmonics coefficients that are stored for a given `l_max`
constexpr SPECTRE_ALWAYS_INLINE size_t
number_of_swsh_coefficients(const size_t l_max) noexcept {
  return (l_max + 1) * (l_max + 2) / 2;  // "triangular" representation
}

/*!
 * \ingroup SwshGroup
 * \brief Compute the relative sign change necessary to convert between the
 * libsharp basis for spin weight `from_spin_weight` to the basis for spin
 * weight `to_spin_weight`, for the real component coefficients if `real` is
 * true, otherwise for the imaginary component coefficients. The sign change for
 * a given coefficient is equivalent to the product of
 * `sharp_swsh_sign(from_spin, m, real) * sharp_swsh_sign(to_spin, m,
 * real)`. Due to the form of the signs, it does not end up depending on m (the
 * m's in the power of \f$-1\f$'s cancel).
 * For full details of the libsharp sign conventions, see the documentation for
 * TransformJob.
 *
 * \details The sign change is obtained by the
 * difference between the libsharp convention and the convention which uses:
 *
 * \f[
 * {}_s Y_{\ell m} (\theta, \phi) = (-1)^m \sqrt{\frac{2 l + 1}{4 \pi}}
 * D^{\ell}{}_{-m s}(\phi, \theta, 0).
 * \f]
 *
 * See \cite Goldberg1966uu
 */
constexpr SPECTRE_ALWAYS_INLINE double sharp_swsh_sign_change(
    const int from_spin_weight, const int to_spin_weight,
    const bool real) noexcept {
  if (real) {
    return (from_spin_weight == 0 ? -1.0 : 1.0) *
           (from_spin_weight >= 0 ? -1.0 : 1.0) *
           ((from_spin_weight < 0 and (from_spin_weight % 2 == 0)) ? -1.0
                                                                   : 1.0) *
           (to_spin_weight == 0 ? -1.0 : 1.0) *
           (to_spin_weight >= 0 ? -1.0 : 1.0) *
           ((to_spin_weight < 0 and (to_spin_weight % 2 == 0)) ? -1.0 : 1.0);
  }
  return (from_spin_weight == 0 ? -1.0 : 1.0) *
         ((from_spin_weight < 0 and (from_spin_weight % 2 == 0)) ? 1.0 : -1.0) *
         (to_spin_weight == 0 ? -1.0 : 1.0) *
         ((to_spin_weight < 0 and (to_spin_weight % 2 == 0)) ? 1.0 : -1.0);
}

/*!
 * \ingroup SwshGroup
 * \brief Compute the sign change between the libsharp convention and the set
 * of spin-weighted spherical harmonics given by the relation to the Wigner
 * rotation matrices.
 *
 * \details The sign change is obtained via the difference between the libsharp
 * convention and the convention which uses:
 *
 * \f[
 * {}_s Y_{\ell m} (\theta, \phi) = (-1)^m \sqrt{\frac{2 l + 1}{4 \pi}}
 * D^{\ell}{}_{-m s}(\phi, \theta, 0).
 * \f]
 *
 * See \cite Goldberg1966uu.
 * The sign change is computed for the real component coefficients if `real` is
 * true, otherwise for the imaginary component coefficients. For full details on
 * the sign convention used in libsharp, see the documentation for TransformJob.
 * This function outputs the \f$\mathrm{sign}(m, s, \mathrm{real})\f$ necessary
 * to produce the conversion between Goldberg moments and libsharp moments:
 *
 * \f{align*}{
 * {}_s Y_{\ell m}^{\mathrm{real}, \mathrm{sharp}}  =& \mathrm{sign}(m, s,
 * \mathrm{real=true}) {}_s Y_{\ell m}^{\mathrm{Goldberg}}\\
 * {}_s Y_{\ell m}^{\mathrm{imag}, \mathrm{sharp}}  =&
 * \mathrm{sign}(m, s, \mathrm{real=false}) {}_s Y_{\ell
 * m}^{\mathrm{Goldberg}}.
 * \f}
 *
 * Note that in this equation, the "real" and "imag" superscripts refer to the
 * set of basis functions used for the decomposition of the real and imaginary
 * part of the spin-weighted collocation points, not real or imaginary parts of
 * the basis functions themselves.
 */
constexpr SPECTRE_ALWAYS_INLINE double sharp_swsh_sign(
    const int spin_weight, const int m, const bool real) noexcept {
  if (real) {
    if (m >= 0) {
      return (spin_weight > 0 ? -1.0 : 1.0) *
             ((spin_weight < 0 and (spin_weight % 2 == 0)) ? -1.0 : 1.0);
    }
    return (spin_weight == 0 ? -1.0 : 1.0) *
           ((spin_weight >= 0 and (m % 2 == 0)) ? -1.0 : 1.0) *
           (spin_weight < 0 and (m + spin_weight) % 2 == 0 ? -1.0 : 1.0);
  }
  if (m >= 0) {
    return (spin_weight == 0 ? -1.0 : 1.0) *
           ((spin_weight < 0 and (spin_weight % 2 == 0)) ? 1.0 : -1.0);
  }
  return (spin_weight == 0 ? -1.0 : 1.0) *
         ((spin_weight >= 0 and (m % 2 == 0)) ? -1.0 : 1.0) *
         ((spin_weight < 0 and (m + spin_weight) % 2 != 0) ? -1.0 : 1.0);
}

namespace detail {

constexpr size_t coefficients_maximum_l_max = collocation_maximum_l_max;

struct DestroySharpAlm {
  void operator()(sharp_alm_info* to_delete) noexcept {
    sharp_destroy_alm_info(to_delete);
  }
};
// The Coefficients class acts largely as a memory-safe container for a
// `sharp_alm_info*`, required for use of libsharp transform utilities.
// The libsharp utilities are currently constructed to only provide user
// functions with collocation data for spin-weighted functions and
// derivatives. If and when the libsharp utilities are expanded to provide
// spin-weighted coefficients as output, this class should be expanded to
// provide information about the value and storage ordering of those
// coefficients to user code. This should be implemented as an iterator, as is
// done in SwshCollocation.hpp.
//
// Note: The libsharp representation of coefficients is altered from the
// standard mathematical definitions in a nontrivial way. There are a number of
// important features to the data storage of the coefficients.
// - they are stored as a set of complex values, but each vector of complex
//   values is the transform of only the real or imaginary part of the
//   collocation data
// - because each vector of complex coefficients is related to the transform of
//   a set of doubles, only (about) half of the m's are stored (m >= 0), because
//   the remaining m modes are determinable by conjugation from the positive m
//   modes, given that they represent the transform of a purely real or purely
//   imaginary collocation quantity
// - they are stored in an l-varies-fastest, triangular representation. To be
//   concrete, for an l_max=2, the order of coefficient storage is (l, m):
//   [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2)]
// - due to the restriction of representing only the transform of real
//   quantities, the m=0 modes always have vanishing imaginary component.
class Coefficients {
 public:
  explicit Coefficients(size_t l_max) noexcept;

  ~Coefficients() = default;
  Coefficients() = default;
  Coefficients(const Coefficients&) = delete;
  Coefficients(Coefficients&&) = default;
  Coefficients& operator=(const Coefficients&) = delete;
  Coefficients& operator=(Coefficients&&) = default;
  sharp_alm_info* get_sharp_alm_info() const noexcept {
    return alm_info_.get();
  }

  size_t l_max() const noexcept { return l_max_; }

 private:
  std::unique_ptr<sharp_alm_info, DestroySharpAlm> alm_info_;
  size_t l_max_ = 0;
};

// Function for obtaining a `Coefficients`, which is a thin wrapper around
// the libsharp `alm_info`, needed to perform transformations and iterate over
// coefficients. A lazy static cache is used to avoid repeated computation. See
// the similar implementation in `SwshCollocation.hpp` for details about the
// caching mechanism.
const Coefficients& precomputed_coefficients(size_t l_max) noexcept;
}  // namespace detail
}  // namespace Swsh
}  // namespace Spectral
