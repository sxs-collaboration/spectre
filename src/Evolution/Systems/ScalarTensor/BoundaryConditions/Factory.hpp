// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/AnalyticConstant.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryConditions/ProductOfConditions.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarTensor::BoundaryConditions {

namespace detail {

// Remove boundary condition products of DemandOutgoingCharSpeeds with
// other types of boundary conditions
template <typename DerivedGhCondition, typename DerivedScalarCondition>
using ProductOfConditionsIfConsistent = tmpl::conditional_t<
    (DerivedGhCondition::bc_type ==
     evolution::BoundaryConditions::Type::DemandOutgoingCharSpeeds) xor
        (DerivedScalarCondition::bc_type ==
         evolution::BoundaryConditions::Type::DemandOutgoingCharSpeeds),
    tmpl::list<>,
    ProductOfConditions<DerivedGhCondition, DerivedScalarCondition>>;

template <typename GhList, typename ScalarList>
struct AllProductConditions;

template <typename GhList, typename... ScalarConditions>
struct AllProductConditions<GhList, tmpl::list<ScalarConditions...>> {
  using type = tmpl::flatten<tmpl::list<tmpl::transform<
      GhList, tmpl::bind<ProductOfConditionsIfConsistent, tmpl::_1,
                         tmpl::pin<ScalarConditions>>>...>>;
};

}  // namespace detail

/// Typelist of standard BoundaryConditions. For now, we only support a subset
/// of the available boundary conditions
using subset_standard_boundary_conditions_gh =
    tmpl::list<gh::BoundaryConditions::DemandOutgoingCharSpeeds<3>,
               gh::BoundaryConditions::DirichletAnalytic<3>>;

using subset_standard_boundary_conditions_scalar = tmpl::list<
    CurvedScalarWave::BoundaryConditions::AnalyticConstant<3>,
    CurvedScalarWave::BoundaryConditions::DemandOutgoingCharSpeeds<3>>;
using standard_boundary_conditions =
    tmpl::push_back<typename detail::AllProductConditions<
                        subset_standard_boundary_conditions_gh,
                        subset_standard_boundary_conditions_scalar>::type,
                    domain::BoundaryConditions::Periodic<BoundaryCondition>>;

}  // namespace ScalarTensor::BoundaryConditions
