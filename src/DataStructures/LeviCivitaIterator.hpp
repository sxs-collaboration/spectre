// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Index.hpp"
#include "Utilities/ConstantExpressions.hpp"

/*!
 * \ingroup DataStructuresGroup
 *
 * \brief Iterate over all nonzero index permutations for a Levi-Civita
 * symbol.
 *
 * \details This class provides an iterator that allows you to loop over
 * only the nonzero index permutations of a Levi-Civita symbol of dimension
 * `dimension`. Inside the loop, the operator `()`
 * returns an `Index` containing an ordered list of the indexes of this
 * permutation, and the function `sign()` returns the sign of the Levi-Civita
 * symbol for that permutation.
 *
 * \example
 * \code
 *   for(LeviCivitaIterator<3> it; it; ++it) {
 *      it(); // ordered list of indexes for this permutation
 *      it.sign(); // sign of Levi-Citivta symbol for this permutation
 *   }
 * \endcode
 */
template <std::size_t Dim>
class LeviCivitaIterator {
 public:
  explicit LeviCivitaIterator() noexcept;

  /// \cond HIDDEN_SYMBOLS
  ~LeviCivitaIterator() = default;
  // @{
  /// No copy or move semantics
  LeviCivitaIterator(const LeviCivitaIterator<Dim>&) = delete;
  LeviCivitaIterator(LeviCivitaIterator<Dim>&&) = delete;
  LeviCivitaIterator<Dim>& operator=(const LeviCivitaIterator<Dim>&) = delete;
  LeviCivitaIterator<Dim>& operator=(LeviCivitaIterator<Dim>&&) = delete;
  // @}
  /// \endcond

  // Return false if the end of the loop is reached
  explicit operator bool() const noexcept { return valid_; }

  // Increment the current permutation
  LeviCivitaIterator& operator++() noexcept {
    ++permutation_;
    if (permutation_ < signs_.size()) {
      valid_ = false;
    }
    return *this;
  }

  // Return the current (multi-index) Index, an ordered list of
  // indices
  const Index<Dim>& operator()() const noexcept {
    return indexes_[permutation_];
  }

  // Return the sign of the Levi-Civita symbol with these indices
  int sign() const noexcept { return signs_[permutation_]; }

 private:
  std::array<Index<Dim>, factorial(Dim + 1)> indexes_;
  std::array<int, factorial(Dim + 1)> signs_;
  size_t permutation_;
  bool valid_;
};
