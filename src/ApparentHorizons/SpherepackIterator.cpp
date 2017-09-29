// Distributed under the MIT License.
// See LICENSE.txt for details.

//
// SPHEREPACK memory layout
//
// For a surface of topology S2 with a spherical harmonic basis and a
// grid of n_th x n_ph points, the largest (partially) representable
// spherical-harmonic Ylm modes (l,m) are:
// * l_max = n_th-1
// * m_max = min(l_max, n_ph/2) [integer division]
//
// Given collocation values on this n_th x n_ph grid, SPHEREPACK
// stores the spectral coefficients in an array of length
// (l_max+1)*(m_max+1).  The layout of this array is, in a
// 2-dimensional representation
//
//                            m_max+1
//     -----------------------------------------------------
//     | (0,0)         x       x            x
//     | (1,0)       (1,1)     x            x         ARRAY
//     | (2,0)       (2,1)   (2,2)          x          "a"
//     |   :           :       :            x
//     | (m_max,  0)  ...            (m_max,   m_max)
//     | (m_max+1,0)  ...            (m_max+1, m_max)
//     |    :                  :            :
//     | (l_max, 0)                  (l_max,   m_max)
//     -----------------------------------------------------
//     |    x        x          x          x
//     |    x      (1,1)        x          x          ARRAY
//     |    x      (2,1)      (2,2)        x           "b"
//     |    :        :          :          x
//     |    x    (m_max,  1)..      (m_max,   m_max)
//     |    x    (m_max+1,1)..      (m_max+1, m_max)
//     |    :         :               :
//     |    x    (l_max,  1)..      (l_max,   m_max)
//     -----------------------------------------------------
//
//
// In this table,
//
// * There are 2*n_th rows and m_max+1 columns. The SPHEREPACK array
//   stores this data "row by row", i.e. the numbers across a row vary
//   fastest.
//
// * There are no entries with negative m. SPHEREPACK uses positive m
//   only, but with complex spectral coefficients. The portions "a"
//   and "b" of the SPHEREPACK array hold the real and imaginary parts
//   of the spectral coefficients.
//
// * Each "x" stands for an entry that is not referenced by SPHEREPACK
//   (so SPHEREPACK wastes some storage space).  There is also one
//   complete unreferenced row in the middle: the first row of the "b"
//   array.
//
// Note that SPHEREPACK accepts arbitrary values of n_th and n_ph as
// input, and then computes the corresponding l_max, m_max.  However,
// some combinations of n_th and n_ph have strange properties:
// * The maximum M that is AT LEAST PARTIALLY represented by (n_th,n_ph)
//   is std::min(n_th-1,n_ph/2). This is called m_max here. But an arbitrary
//   (n_th,n_ph) does not necessarily fully represent all m's up to m_max,
//   because sin(m_max phi) might be zero at all collocation points, and
//   therefore sin(m_max phi) might not be representable on the grid.
// * The largest m that is fully represented by (n_th,n_ph) is
//   m_max_represented = std::min(n_th-1,(n_ph-1)/2).
// * Therefore, if n_ph is odd,  m_max = m_max_represented,
//              if n_ph is even, m_max = m_max_represented+1.
// * To fully represent a desired (l_max, m_max), the grid resolution
//   should satisfy
//     n_th = l_max+1
//     n_ph = 2*m_max_represented+1
//   which ensures that m_max = m_max_represented.

#include "ApparentHorizons/SpherepackIterator.hpp"

SpherepackIterator::SpherepackIterator(const size_t l_max_input,
                                       const size_t m_max_input,
                                       const size_t stride /*=1*/)
    : l_max_(l_max_input),
      m_max_(m_max_input),
      n_th_(l_max_ + 1),
      n_ph_(2 * m_max_ + 1),
      stride_(stride),
      number_of_valid_entries_in_a_((m_max_ + 1) * (m_max_ + 2) / 2 +
                                    (m_max_ + 1) * (l_max_ - m_max_)),
      current_compact_index_(0) {
  // fill offset_into_spherepack_array, compact_l_ and compact_m_ with the
  // indices, l-values and m-values of all valid points
  const size_t packed_size =
      (m_max_ + 1) * (m_max_ + 2) / 2 + m_max_ * (m_max_ + 1) / 2 +
      (m_max_ + 1) * (l_max_ - m_max_) + m_max_ * (l_max_ - m_max_);

  offset_into_spherepack_array.assign(packed_size, 0);
  compact_l_.assign(packed_size, 0);
  compact_m_.assign(packed_size, 0);

  // go through arrays and fill them

  // index corresponding to strided coefficient array
  size_t idx = 0;
  // index for compact offset_into_spherepack_array, compact_l_, compact_m_
  size_t k = 0;
  for (size_t l = 0; l <= l_max_; ++l) {
    for (size_t m = 0; m <= m_max_; ++m) {
      // note: index m varies fastest in fortran array
      if (l >= m) {  // valid entry in a
        offset_into_spherepack_array[k] = idx;
        compact_l_[k] = l;
        compact_m_[k] = m;
        ++k;
      }
      idx += stride;
    }
  }
  for (size_t l = 0; l <= l_max_; ++l) {
    for (size_t m = 0; m <= m_max_; ++m) {
      // note: index m varies fastest in fortran array
      if (m >= 1 && l >= m) {  // valid entry in b
        offset_into_spherepack_array[k] = idx;
        compact_l_[k] = l;
        compact_m_[k] = m;
        ++k;
      }
      idx += stride;
    }
  }
}

SpherepackIterator& SpherepackIterator::reset() noexcept {
  current_compact_index_ = 0;
  return *this;
}

SpherepackIterator& SpherepackIterator::set(
    const size_t l_input, const size_t m_input,
    const CoefficientArray coefficient_array_input) noexcept {
  ASSERT(l_input <= l_max_, "SpherepackIterator l_input="
                                << l_input
                                << " too large, should be <= " << l_max_);
  ASSERT(l_input >= m_input, "SpherepackIterator m_input "
                                 << m_input << " greater than l_input "
                                 << l_input);
  ASSERT(m_input <= m_max_,
         "m_input " << m_input << " too large, should be <= " << m_max_);
  if (coefficient_array_input == CoefficientArray::a) {
    // jump to first element with 'l_input'
    if (l_input <= m_max_ + 1) {
      current_compact_index_ = (l_input * (l_input + 1)) / 2;
    } else {
      current_compact_index_ = (m_max_ + 1) * (m_max_ + 2) / 2 +
                               (l_input - m_max_ - 1) * (m_max_ + 1);
    }
    current_compact_index_ += m_input;
  } else {
    ASSERT(m_input != 0, "Array b does not contain m_input=0");
    if (l_input <= m_max_ + 1) {
      current_compact_index_ = (l_input * (l_input - 1)) / 2;
    } else {
      current_compact_index_ =
          m_max_ * (m_max_ + 1) / 2 + (l_input - m_max_ - 1) * m_max_;
    }
    current_compact_index_ += number_of_valid_entries_in_a_ + m_input - 1;
  }

  ASSERT(current_compact_index_ < offset_into_spherepack_array.size(),
         "Expected " << current_compact_index_ << " to be less than "
                     << offset_into_spherepack_array.size());
  ASSERT(l_input == compact_l_[current_compact_index_],
         "Expected " << l_input << " but set produced "
                     << compact_l_[current_compact_index_]);
  ASSERT(m_input == compact_m_[current_compact_index_],
         "Expected " << m_input << " but set produced "
                     << compact_m_[current_compact_index_]);

  return *this;
}
