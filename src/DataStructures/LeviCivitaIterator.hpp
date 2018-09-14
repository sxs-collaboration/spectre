// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

#include "Utilities/Algorithm.hpp"
#include "Utilities/Array.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Numeric.hpp"

/*!
 * \ingroup DataStructuresGroup
 *
 * \brief Iterate over all nonzero index permutations for a Levi-Civita
 * symbol.
 *
 * \details This class provides an iterator that allows you to loop over
 * only the nonzero index permutations of a Levi-Civita symbol of dimension
 * `dimension`. Inside the loop, the operator `()`
 * returns an `std::array` containing an ordered list of the indices of this
 * permutation, the operator `[]` returns a specific index from the same
 * `std::array`, and the function `sign()` returns the sign of the
 * Levi-Civita symbol for this permutation.
 *
 * \example
 * \snippet Test_LeviCivitaIterator.cpp levi_civita_iterator_example
 */
template <size_t Dim>
class LeviCivitaIterator {
 public:
  explicit LeviCivitaIterator() noexcept = default;

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
    if (permutation_ >= signs_.size()) {
      valid_ = false;
    }
    return *this;
  }

  /// Return a `std::array` containing the current multi-index, an ordered list
  /// of indices for the current permutation.
  const std::array<size_t, Dim> operator()() const noexcept {
    return static_cast<std::array<size_t, Dim>>(indexes_[permutation_]);
  }

  /// Return a specific index from the multi-index of the current permutation.
  const size_t& operator[](const size_t i) const noexcept {
    return indexes_[permutation_][i];
  }

  /// Return the sign of the Levi-Civita symbol for the current permutation.
  int sign() const noexcept { return signs_[permutation_]; };

 private:
  static constexpr cpp17::array<cpp17::array<size_t, Dim>, factorial(Dim)>
  indexes() noexcept {
    cpp17::array<cpp17::array<size_t, Dim>, factorial(Dim)> indexes{};
    cpp17::array<size_t, Dim> index{};
    cpp2b::iota(index.begin(), index.end(), size_t(0));

    indexes[0] = index;

    // cpp20::next_permutation generates the different permuations of index
    // loop over them to fill in the rest of the permutations in indexes
    size_t permutation = 1;
    while (cpp20::next_permutation(index.begin(), index.end())) {
      indexes[permutation] = index;
      ++permutation;
    }
    return indexes;
  };

  static constexpr cpp17::array<int, factorial(Dim)> signs() noexcept {
    cpp17::array<int, factorial(Dim)> signs{};
    // By construction, the sign of the first permutation is +1
    signs[0] = 1;

    // How do you know whether the corresponding Levi Civita symbol is
    // +1 or -1? To find out, compute the number, in the factoradic number
    // system, corresponding to each permutation, and sum the digits. If the sum
    // is even (odd), the corresponding Levi-Civita symbol will be +1 (-1). This
    // works because there are Dim! unique permutations of the Levi Civita
    // symbol indices, so each one can be represented by a
    // (`Dim`)-digit factorial-number-system number. For more on the
    // factoradic number system, see
    // https://en.wikipedia.org/wiki/Factorial_number_system
    auto factoradic_counter = cpp17::array<size_t, Dim>();
    for (size_t permutation = 1; permutation < factorial(Dim); ++permutation) {
      for (size_t i = 0; i < Dim; ++i) {
        factoradic_counter[i]++;
        if (factoradic_counter[i] < i + 1) {
          break;
        } else {
          factoradic_counter[i] = 0;
        }
      }
      signs[permutation] =
          (cpp2b::accumulate(factoradic_counter.begin(),
                             factoradic_counter.end(), size_t(0)) %
           2) == 0
              ? 1
              : -1;
    }
    return signs;
  };

  // Note: here and throughout, use cpp17::array,
  // which is constexpr (unlike std::array), to enable constexpr signs_
  // and indexes_.
  static constexpr cpp17::array<cpp17::array<size_t, Dim>, factorial(Dim)>
      indexes_ = indexes();
  static constexpr cpp17::array<int, factorial(Dim)> signs_{signs()};
  size_t permutation_{0};
  bool valid_{true};
};
