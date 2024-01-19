// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Imex/SolveImplicitSector.hpp"
#include "Evolution/Imex/SolveImplicitSector.tpp"
#include "Helpers/Evolution/Imex/DoImplicitStepSector.hpp"

namespace helpers = do_implicit_step_helpers;
template struct imex::SolveImplicitSector<helpers::System::variables_tag,
                                          helpers::Sector<helpers::Var1>>;
template struct imex::SolveImplicitSector<helpers::System::variables_tag,
                                          helpers::Sector<helpers::Var2>>;
template struct imex::SolveImplicitSector<
    helpers::NonautonomousSystem::variables_tag, helpers::NonautonomousSector>;
