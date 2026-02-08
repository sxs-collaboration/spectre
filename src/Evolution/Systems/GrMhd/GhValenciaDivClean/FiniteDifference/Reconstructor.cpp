// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/NeutrinoSystems.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"

#include <pup.h>

#include "Utilities/GenerateInstantiations.hpp"
namespace grmhd::GhValenciaDivClean::fd {
template <typename System>
void Reconstructor<System>::pup([[maybe_unused]] PUP::er& p) {
#if defined(SPECTRE_USE_CHARM)
  PUP::able::pup(p);
#endif  // SPECTRE_USE_CHARM
}

#define NEUTRINO(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)  \
  template class Reconstructor< \
      typename grmhd::GhValenciaDivClean::System<NEUTRINO(data)>>;

GENERATE_INSTANTIATIONS(INSTANTIATION, GHMHD_NEUTRINOS)

#undef INSTANTIATION
#undef NEUTRINO

}  // namespace grmhd::GhValenciaDivClean::fd
