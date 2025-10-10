// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryCorrections/ProductOfCorrections.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarTensor::BoundaryCorrections {
namespace detail {
template <typename GhList, typename ScalarList>
struct AllProductCorrections;

template <typename GhList, typename... ScalarCorrections>
struct AllProductCorrections<GhList, tmpl::list<ScalarCorrections...>> {
  using type = tmpl::flatten<tmpl::list<
      tmpl::transform<GhList, tmpl::bind<ProductOfCorrections, tmpl::_1,
                                         tmpl::pin<ScalarCorrections>>>...>>;
};
}  // namespace detail

using standard_boundary_corrections = typename detail::AllProductCorrections<
    typename gh::BoundaryCorrections::standard_boundary_corrections<3>,
    typename CurvedScalarWave::BoundaryCorrections::
        standard_boundary_corrections<3>>::type;
}  // namespace ScalarTensor::BoundaryCorrections
