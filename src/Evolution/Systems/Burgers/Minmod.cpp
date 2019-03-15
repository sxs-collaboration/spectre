// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/Minmod.tpp"  // IWYU pragma: keep
#include "Evolution/Systems/Burgers/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include "Evolution/DiscontinuousGalerkin/SlopeLimiters/Minmod.hpp"

template class SlopeLimiters::Minmod<1, tmpl::list<Burgers::Tags::U>>;
