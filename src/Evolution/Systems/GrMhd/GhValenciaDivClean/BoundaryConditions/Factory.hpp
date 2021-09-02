// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/ProductOfConditions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::BoundaryConditions {
namespace detail {
template <typename DerivedGhCondition, typename DerivedValenciaCondition>
using ProductOfConditionsIfConsistent = tmpl::conditional_t<
    (DerivedGhCondition::bc_type ==
     evolution::BoundaryConditions::Type::Outflow) xor
        (DerivedValenciaCondition::bc_type ==
         evolution::BoundaryConditions::Type::Outflow),
    tmpl::list<>,
    ProductOfConditions<DerivedGhCondition, DerivedValenciaCondition>>;

template <typename GhList, typename ValenciaList>
struct AllProductConditions;

template <typename GhList, typename... ValenciaConditions>
struct AllProductConditions<GhList, tmpl::list<ValenciaConditions...>> {
  using type = tmpl::flatten<tmpl::list<tmpl::transform<
      GhList, tmpl::bind<ProductOfConditionsIfConsistent, tmpl::_1,
                         tmpl::pin<ValenciaConditions>>>...>>;
};

template <typename ClassList>
struct remove_periodic_conditions {
  using type = tmpl::remove_if<
      ClassList,
      std::is_base_of<domain::BoundaryConditions::MarkAsPeriodic, tmpl::_1>>;
};

template <typename ClassList>
using remove_periodic_conditions_t =
    typename remove_periodic_conditions<ClassList>::type;
}  // namespace detail

// remove the periodic BCs from the creatable classes of the
// individual systems; for the remaining conditions, include a
// `ProductOfConditions` for each pair with compatible `bc_type`s.
/// Typelist of standard BoundaryConditions
using standard_boundary_conditions = tmpl::push_back<
    typename detail::AllProductConditions<
        detail::remove_periodic_conditions_t<
            typename GeneralizedHarmonic::BoundaryConditions::
                standard_boundary_conditions<3_st>>,
        detail::remove_periodic_conditions_t<
            typename grmhd::ValenciaDivClean::BoundaryConditions::
                standard_boundary_conditions>>::type,
    domain::BoundaryConditions::Periodic<BoundaryCondition>>;
}  // namespace grmhd::GhValenciaDivClean::BoundaryConditions
