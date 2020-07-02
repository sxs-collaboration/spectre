// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.tpp"  // IWYU pragma: keep
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"

namespace Frame {
struct Inertial;
}  // namespace Frame

// Do not limit the divergence-cleaning field Phi
template class Limiters::Minmod<
    3, tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
                  grmhd::ValenciaDivClean::Tags::TildeTau,
                  grmhd::ValenciaDivClean::Tags::TildeS<>,
                  grmhd::ValenciaDivClean::Tags::TildeB<>>>;
