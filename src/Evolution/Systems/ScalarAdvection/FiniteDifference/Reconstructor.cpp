// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Reconstructor.hpp"

#include <cstddef>
#include <pup.h>

#include "Utilities/GenerateInstantiations.hpp"

namespace ScalarAdvection::fd {
template <size_t Dim>
Reconstructor<Dim>::Reconstructor(CkMigrateMessage* const msg) noexcept
    : PUP::able(msg) {}

template <size_t Dim>
void Reconstructor<Dim>::pup(PUP::er& p) {
  PUP::able::pup(p);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data) template class Reconstructor<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))

#undef INSTANTIATION
#undef DIM
}  // namespace ScalarAdvection::fd
