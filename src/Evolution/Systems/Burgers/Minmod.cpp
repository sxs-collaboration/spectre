// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.tpp"  // IWYU pragma: keep
#include "Evolution/Systems/Burgers/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"

template class Limiters::Minmod<1, tmpl::list<Burgers::Tags::U>>;
