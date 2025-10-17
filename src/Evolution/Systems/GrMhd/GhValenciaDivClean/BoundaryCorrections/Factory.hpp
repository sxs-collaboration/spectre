// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/ProductOfCorrections.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::BoundaryCorrections {
namespace detail {
template <typename GhList, typename ValenciaList>
struct AllProductCorrections;

template <typename GhList, typename... ValenciaCorrections>
struct AllProductCorrections<GhList, tmpl::list<ValenciaCorrections...>> {
  using type = tmpl::flatten<tmpl::list<
      tmpl::transform<GhList, tmpl::bind<ProductOfCorrections, tmpl::_1,
                                         tmpl::pin<ValenciaCorrections>>>...>>;
};
}  // namespace detail

using standard_boundary_corrections = typename detail::AllProductCorrections<
    typename gh::BoundaryCorrections::standard_boundary_corrections<3_st>,
    typename grmhd::ValenciaDivClean::BoundaryCorrections::
        standard_boundary_corrections>::type;
}  // namespace grmhd::GhValenciaDivClean::BoundaryCorrections
