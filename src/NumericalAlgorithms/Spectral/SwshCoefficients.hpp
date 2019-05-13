// Distributed under the MIT License.
// See LICENSE.txt for details

#pragma once

#include <cstdlib>
#include <memory>
#include <sharp_cxx.h>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "Utilities/ForceInline.hpp"

namespace Spectral {
namespace Swsh {

/// \ingroup SwshGroup
/// \brief Convenience function for determining the number of spin-weighted
/// spherical harmonics coefficients that are stored for a given `l_max`
///
/// \details This includes the factor of 2 associated with needing
/// to store both the transform of the real and imaginary parts, so is the
/// full size of the result of a libsharp swsh transform.
///
/// \note Assumes the triangular libsharp representation is used.
constexpr SPECTRE_ALWAYS_INLINE size_t
size_of_libsharp_coefficient_vector(const size_t l_max) noexcept {
  return (l_max + 1) * (l_max + 2);  // "triangular" representation
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
}  // namespace detail

/// Points to a single pair of modes in a libsharp-compatible
/// spin-weighted spherical harmonic modal representation.
struct LibsharpCoefficientInfo {
  size_t transform_of_real_part_offset;
  size_t transform_of_imag_part_offset;
  size_t l_max;
  size_t l;
  size_t m;
};

/*!
 * \ingroup SwshGroup
 * \brief A container for libsharp metadata for the spin-weighted spherical
 * harmonics modal representation.
 *
 * \details
 * The CoefficientsMetadata class acts as a memory-safe container for a
 * `sharp_alm_info*`, required for use of libsharp transform utilities.
 * The libsharp utilities are currently constructed to only provide user
 * functions with collocation data for spin-weighted functions and
 * derivatives.  This class also provides an iterator for
 * easily traversing a libsharp-compatible modal representation.
 *
 * \note The libsharp representation of coefficients is altered from the
 * standard mathematical definitions in a nontrivial way. There are a number of
 * important features to the data storage of the coefficients.
 * - they are stored as a set of complex values, but each vector of complex
 *   values is the transform of only the real or imaginary part of the
 *   collocation data
 * - because each vector of complex coefficients is related to the transform of
 *   a set of doubles, only (about) half of the m's are stored (m >= 0), because
 *   the remaining m modes are determinable by conjugation from the positive m
 *   modes, given that they represent the transform of a purely real or purely
 *   imaginary collocation quantity
 * - they are stored in an l-varies-fastest, triangular representation. To be
 *   concrete, for an l_max=2, the order of coefficient storage is (l, m):
 *   [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2)]
 * - due to the restriction of representing only the transform of real
 *   quantities, the m=0 modes always have vanishing imaginary component.
 */
class CoefficientsMetadata {
 public:
  /// An iterator for easily traversing a libsharp-compatible spin-weighted
  /// spherical harmonic modal representation.
  /// The `operator*()` returns a `LibsharpCoefficientInfo`, which  contains two
  /// offsets, `transform_of_real_part_offset` and
  /// `transform_of_imag_part_offset`, and the `l_max`, `l` and `m` associated
  /// with the values at those offsets.
  ///
  /// \note this currently assumes, as do many of the utilities in this file,
  /// that the libsharp representation is chosen to be the triangular
  /// coefficient representation. If alternative representations are desired,
  /// alterations will be needed.
  class CoefficientsIndexIterator {
   public:
    explicit CoefficientsIndexIterator(const size_t l_max,
                                       const size_t start_l = 0,
                                       const size_t start_m = 0) noexcept
        : m_{start_m}, l_{start_l}, l_max_{l_max} {}

    LibsharpCoefficientInfo operator*() const noexcept {
      // permit dereferencing the iterator only if the return represents a
      // viable location in the coefficient vector
      ASSERT(l_ <= l_max_ && m_ <= l_max_, "coefficients iterator overflow");
      size_t offset = ((3 + 2 * l_max_ - m_) * m_) / 2 + l_ - m_;
      return LibsharpCoefficientInfo{
          offset,
          offset +
              Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_) / 2,
          l_max_, l_, m_};
    }

    /// advance the iterator by one position (prefix)
    CoefficientsIndexIterator& operator++() noexcept {
      // permit altering the iterator only if the new value is between the
      // anticipated begin and end, inclusive
      ASSERT(l_ <= l_max_ && m_ <= l_max_, "coefficients iterator overflow");
      if (l_ == l_max_) {
        ++m_;
        l_ = m_;
      } else {
        ++l_;
      }
      return *this;
    };

    /// advance the iterator by one position (postfix)
    const CoefficientsIndexIterator operator++(int) noexcept {
      auto pre_increment = *this;
      ++*this;
      return pre_increment;
    }

    /// retreat the iterator by one position (prefix)
    CoefficientsIndexIterator& operator--() noexcept {
      // permit altering the iterator only if the new value is between the
      // anticipated begin and end, inclusive
      ASSERT(l_ <= l_max_ + 1 && m_ <= l_max_ + 1,
             "coefficients iterator overflow");
      if (l_ == m_) {
        --m_;
        l_ = l_max_;
      } else {
        --l_;
      }
      return *this;
    }

    /// retreat the iterator by one position (postfix)
    const CoefficientsIndexIterator operator--(int) noexcept {
      auto pre_decrement = *this;
      --*this;
      return pre_decrement;
    }

    // @{
    /// (In)Equivalence checks the object as well as the l and m current
    /// position.
    bool operator==(const CoefficientsIndexIterator& rhs) const noexcept {
      return m_ == rhs.m_ and l_ == rhs.l_ and l_max_ == rhs.l_max_;
    }
    bool operator!=(const CoefficientsIndexIterator& rhs) const noexcept {
      return not(*this == rhs);
    }
    // @}

   private:
    size_t m_;
    size_t l_;
    size_t l_max_;
  };

  explicit CoefficientsMetadata(size_t l_max) noexcept;

  ~CoefficientsMetadata() = default;
  CoefficientsMetadata() = default;
  CoefficientsMetadata(const CoefficientsMetadata&) = delete;
  CoefficientsMetadata(CoefficientsMetadata&&) = default;
  CoefficientsMetadata& operator=(const CoefficientsMetadata&) = delete;
  CoefficientsMetadata& operator=(CoefficientsMetadata&&) = default;
  sharp_alm_info* get_sharp_alm_info() const noexcept {
    return alm_info_.get();
  }

  size_t l_max() const noexcept { return l_max_; }

  /// returns the number of (complex) entries in a libsharp-compatible
  /// coefficients vector. This includes the factor of 2 associated with needing
  /// to store both the transform of the real and imaginary parts, so is the
  /// full size of the result of a libsharp swsh transform.
  size_t size() const noexcept {
    return size_of_libsharp_coefficient_vector(l_max_);
  }

  // @{
  /// \brief Get a bidirectional iterator to the start of the series of modes.
  CoefficientsMetadata::CoefficientsIndexIterator begin() const noexcept {
    return CoefficientsIndexIterator(l_max_, 0, 0);
  }
  CoefficientsMetadata::CoefficientsIndexIterator cbegin() const noexcept {
    return begin();
  }
  // @}

  // @{
  /// \brief Get a bidirectional iterator to the end of the series of modes.
  CoefficientsMetadata::CoefficientsIndexIterator end() const noexcept {
    return CoefficientsIndexIterator(l_max_, l_max_ + 1, l_max_ + 1);
  }
  CoefficientsMetadata::CoefficientsIndexIterator cend() const noexcept {
    return end();
  }
  // @}

 private:
  std::unique_ptr<sharp_alm_info, detail::DestroySharpAlm> alm_info_;
  size_t l_max_ = 0;
};

/*!
 * \ingroup SwshGroup
 * \brief Generation function for obtaining a `CoefficientsMetadata` object
 * which is computed by the libsharp calls only once, then lazily cached as a
 * singleton via a static member of a function template. This is the preferred
 * method for obtaining a `CoefficientsMetadata` when the `l_max` is not very
 * large
 *
 * See the comments in the similar implementation found in `SwshCollocation.hpp`
 * for more details on the lazy cache.
 */
const CoefficientsMetadata& cached_coefficients_metadata(size_t l_max) noexcept;
}  // namespace Swsh
}  // namespace Spectral
