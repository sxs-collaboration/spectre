// Distributed under the MIT License.
// See LICENSE.txt for details.

///
/// \file
/// Defines class SpherepackIterator.

#pragma once

#include <cmath>
#include <cstddef>
#include <ostream>
#include <vector>

#include "ErrorHandling/Assert.hpp"

/*!
 * \ingroup SpectralGroup
 * \brief
 * Iterates over spectral coefficients stored in SPHEREPACK format.
 *
 * \details
 * The internal SPHEREPACK ordering is not intuitive, so
 * SpherepackIterator exists for the purpose of iterating over an
 * array containing SPHEREPACK coefficients and determining the
 * (l,m) of each entry in the array.
 *
 * SPHEREPACK expands \f$f(\theta,\phi)\f$ as
 * \f[
 * f(\theta,\phi) =
 * \frac{1}{2} \sum_{l=0}^{l_{max}} \bar{P}_l^0 a(0,l)
 *   + \sum_{m=1}^{m_{max}} \sum_{l=m}^{l_{max}} \bar{P}_l^m
 *         \left(  a(m,l) \cos(m \phi) - b(m,l) \sin(m \phi)\right)
 * \f]
 * where \f$a(m,l)\f$ and \f$b(m,l)\f$ are the SPHEREPACK
 * spectral coefficients, and \f$\bar{P}_l^m\f$ are
 * unit-orthonormal associated Legendre polynomials:
 * \f[
 * \bar{P}_l^m =
 * (-1)^m \sqrt{\frac{(2l+1)(l-m)!}{2(l+m)!}} P_l^m,
 * \f]
 * where \f$P_l^m\f$ are the associated Legendre polynomials as defined
 * for example in Jackson "Classical Electrodynamics".
 * \example
 * \snippet Test_SpherepackIterator.cpp spherepack_iterator_example
 */
class SpherepackIterator {
 public:
  /// SPHEREPACK has two coefficient variables, 'a' and 'b', that hold
  /// the cos(m*phi) and sin(m*phi) parts of the spectral
  /// coefficients.
  enum class CoefficientArray { a, b };

  SpherepackIterator(size_t l_max_input, size_t m_max_input, size_t stride = 1);

  size_t l_max() const noexcept { return l_max_; }
  size_t m_max() const noexcept { return m_max_; }
  size_t n_th() const noexcept { return n_th_; }
  size_t n_ph() const noexcept { return n_ph_; }
  size_t stride() const noexcept { return stride_; }

  /// Size of a SPHEREPACK coefficient array (a and b combined), not
  /// counting stride.  For non-unit stride, the size of the array
  /// should be spherepack_array_size()*stride.
  size_t spherepack_array_size() const noexcept {
    return (l_max_ + 1) * (m_max_ + 1) * 2;
  }

  SpherepackIterator& operator++() noexcept {
    ASSERT(current_compact_index_ != offset_into_spherepack_array.size(),
           "Incrementing an invalid iterator: "
               << current_compact_index_ << " "
               << offset_into_spherepack_array.size());
    ++current_compact_index_;
    return *this;
  }

  explicit operator bool() const noexcept {
    return current_compact_index_ < offset_into_spherepack_array.size();
  }

  /// Current index into a SPHEREPACK coefficient array.
  size_t operator()() const noexcept {
    return offset_into_spherepack_array[current_compact_index_];
  }

  /// Current values of l and m.
  size_t l() const noexcept { return compact_l_[current_compact_index_]; }
  size_t m() const noexcept { return compact_m_[current_compact_index_]; }
  ///  Whether the iterator points to an element of 'a' or 'b',
  ///  i.e. points to the cos(m*phi) or sin(m*phi) part of a spectral
  ///  coefficient.
  CoefficientArray coefficient_array() const noexcept {
    return (current_compact_index_ < number_of_valid_entries_in_a_
                ? CoefficientArray::a
                : CoefficientArray::b);
  }

  /// Reset iterator back to beginning value. Returns *this.
  SpherepackIterator& reset() noexcept;

  /// Set iterator to specific value of l, m, array.  Returns *this.
  SpherepackIterator& set(size_t l_input, size_t m_input,
                          CoefficientArray coefficient_array_input) noexcept;

  /// Same as 'set' above, but assumes CoefficientArray is 'a' for m>=0 and
  /// 'b' for m<0.  This is useful when converting between true
  /// spherical harmonics (which allow negative values of m) and
  /// SPHEREPACK coefficients (which have only positive values of m,
  /// but two arrays for sin(m*phi) and cos(m*phi) parts).
  SpherepackIterator& set(size_t l_input, int m_input) noexcept {
    return set(l_input, size_t(abs(m_input)),
               m_input >= 0 ? CoefficientArray::a : CoefficientArray::b);
  }

 private:
  size_t l_max_, m_max_, n_th_, n_ph_, stride_;
  size_t number_of_valid_entries_in_a_;
  size_t current_compact_index_;
  std::vector<size_t> offset_into_spherepack_array;
  std::vector<size_t> compact_l_, compact_m_;
};

inline bool operator==(const SpherepackIterator& lhs,
                       const SpherepackIterator& rhs) {
  return lhs.l_max() == rhs.l_max() and lhs.m_max() == rhs.m_max() and
         lhs.stride() == rhs.stride() and lhs() == rhs();
}

inline bool operator!=(const SpherepackIterator& lhs,
                       const SpherepackIterator& rhs) {
  return not(lhs == rhs);
}

inline std::ostream& operator<<(
    std::ostream& os,
    const SpherepackIterator::CoefficientArray& coefficient_array) {
  return os << (coefficient_array == SpherepackIterator::CoefficientArray::a
                    ? 'a'
                    : 'b');
}
