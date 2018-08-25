// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/LeviCivitaIterator.hpp"

#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim>
constexpr void initialize_indexes_and_signs(
    gsl::not_null<std::array<Index<Dim>, factorial(Dim + 1)>*> indexes,
    gsl::not_null<std::array<int, factorial(Dim + 1)>*> signs) noexcept {
  auto index = Index<Dim>();
  std::iota(index.begin(), index.end(), 0);

  // Add the initial index permutation to the list
  // Also add the sign, which is always +1, because std::iota
  // initializes the indexes as 0,1,...dim
  (*indexes)[0] = index;
  (*signs)[0] = 1;

  size_t permutation_count = 1;
  auto factoradic_counter = std::array<size_t, Dim>();

  // std::next_permutation generates the different permuations of index,
  // but how do you know whether the corresponding Levi Civita symbol is
  // +1 or -1? To find out, compute the number, in the factorial number system,
  // corresponding to each permutation, and sum the digits. If the sum is
  // even (odd), the corresponding Levi-Civita symbol will be +1 (-1).
  // This works because there are Dim! unique permutations of the
  // Levi Civita symbol indices, so each one can be represented by a
  // (dim+1)-digit factorial-number-system number. For more on the
  // factorial number system, see
  // https://en.wikipedia.org/wiki/Factorial_number_system
  for (; std::next_permutation(index.begin(), index.end());) {
    for (size_t i = 0; i < Dim; ++i) {
      factoradic_counter[i]++;
      if (factoradic_counter[i] < i + 1) {
        break;
      } else {
        factoradic_counter[i] = 0;
      }
    }

    (*indexes)[permutation_count] = index;
    (*signs)[permutation_count] =
        ((std::accumulate(factoradic_counter.begin(), factoradic_counter.end(),
                          0) %
          2) == 0)
            ? 1
            : -1;
    ++permutation_count;
  }
}
}  // namespace

template <size_t Dim>
LeviCivitaIterator<Dim>::LeviCivitaIterator() noexcept
    : permutation_(0), valid_(true) {
  initialize_indexes_and_signs<Dim>(&indexes_, &signs_);
}

template class LeviCivitaIterator<2>;
template class LeviCivitaIterator<3>;
template class LeviCivitaIterator<4>;
