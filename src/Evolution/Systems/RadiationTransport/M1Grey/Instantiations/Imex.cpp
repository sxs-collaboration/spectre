// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Imex/SolveImplicitSector.hpp"
#include "Evolution/Imex/SolveImplicitSector.tpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/System.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"
#include "Utilities/TMPL.hpp"

using m1system = RadiationTransport::M1Grey::System<
    tmpl::list<neutrinos::ElectronNeutrinos<1>>>;

template struct imex::SolveImplicitSector<
    m1system::variables_tag,
    m1system::ImplicitSector<neutrinos::ElectronNeutrinos<1>>>;
