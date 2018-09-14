// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/LeviCivitaIterator.hpp"

template <size_t Dim>
constexpr cpp17::array<cpp17::array<size_t, Dim>, factorial(Dim)>
    LeviCivitaIterator<Dim>::indexes_;

template <size_t Dim>
constexpr cpp17::array<int, factorial(Dim)> LeviCivitaIterator<Dim>::signs_;

// We don't expect to use the Dim==0 case, so only instantiate Dim==1,2,3,4
template class LeviCivitaIterator<1>;
template class LeviCivitaIterator<2>;
template class LeviCivitaIterator<3>;
template class LeviCivitaIterator<4>;
