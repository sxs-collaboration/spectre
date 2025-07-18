// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

/*!
 * \brief Computes Wigner 3J symbols.
 *
 * Given $l_2$, $m_2$, $l_3$, and $m_3$,
 * computes and caches the Wigner 3J symbols
 *
 * \begin{align}
 * \left(
 * \begin{array}{ccc}
 *  l_1 & l_2 & l_3 \\
 * -m_3-m_2 & m_2 & m_3
 * \end{array}\right)
 * \end{align}
 *
 * for all $l_1$.
 *
 * Internally, uses recursion relations to efficiently compute many 3J symbols
 * at once.  This should be more efficient than having a single function
 * that takes $l_2$, $m_2$, $l_3$, $m_3$, $l_1$
 * and computes a single 3J symbol, assuming
 * that the user creates WignerThreeJ objects infrequently, and that the user
 * calls each WignerThreeJ object multiple times.
 *
 * Here we limit the quantum numbers to integer values.  It is trivial to
 * extend to half-integer values because the underlying routine (the
 * drc3jj.f function from the slatec library) supports it.
 *
 * Uses lazy evaluation: the constructor computes nothing, but the first time
 * operator() is called for a nonzero 3J symbol, it computes all the
 * nonzero 3J symbols.
 *
 * Internally, allocates a std::vector.  A future optimization may be to
 * pass in a buffer of the appropriate size, so that WignerThreeJ would not
 * need to do memory allocations.
 */
class WignerThreeJ {
 public:
  WignerThreeJ(size_t l2, int m2, size_t l3, int m3);
  /*!
   * Returns the Wigner 3J symbol
   * \begin{align}
   * \left(
   * \begin{array}{ccc}
   *  l_1 & l_2 & l_3 \\
   * -m_3-m_2 & m_2 & m_3
   * \end{array}\right).
   * \end{align}
   *
   * Internally, computes and caches all nonzero 3J symbols for all $l_1$.
   * Returns zero for $l_1 < l_1^\mathrm{min}$ and
   * $l_1 > l_1^\mathrm{max}$.
   */
  double operator()(size_t l1);
  /// Returns the smallest value of $l_1$ that gives a nonzero 3J symbol.
  size_t l1_min() const { return l1_min_; }
  /// Returns the largest value of $l_1$ that gives a nonzero 3J symbol.
  size_t l1_max() const { return l1_max_; }

 private:
  void recompute();
  // The underlying routine takes doubles (to deal with half-integer spins),
  // so we need to store doubles here.
  double l2_, m2_, l3_, m3_;
  size_t l1_min_, l1_max_;
  bool up_to_date_;
  std::vector<double> coefs_;
};
