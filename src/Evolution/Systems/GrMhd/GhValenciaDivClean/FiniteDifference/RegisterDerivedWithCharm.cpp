// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/RegisterDerivedWithCharm.hpp"

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace grmhd::GhValenciaDivClean::fd {

template <typename System>
void register_derived_with_charm() {
  register_classes_with_charm(
      typename Reconstructor<System>::creatable_classes{});
}

#define NEUTRINO(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template void                \
  register_derived_with_charm<GhValenciaDivClean::System<NEUTRINO(data)>>();

GENERATE_INSTANTIATIONS(INSTANTIATION,
                        (RadiationTransport::NoNeutrinos::System))

#undef INSTANTIATION
#undef NEUTRINO

}  // namespace grmhd::GhValenciaDivClean::fd
