// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/CleanMortarHistory.tpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"
#include "Utilities/GenerateInstantiations.hpp"

#define NEUTRINO(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                      \
  template class evolution::dg::CleanMortarHistory< \
      grmhd::GhValenciaDivClean::System<NEUTRINO(data)>>;

GENERATE_INSTANTIATIONS(INSTANTIATION,
                        (RadiationTransport::NoNeutrinos::System))

#undef INSTANTIATION
#undef NEUTRINO
