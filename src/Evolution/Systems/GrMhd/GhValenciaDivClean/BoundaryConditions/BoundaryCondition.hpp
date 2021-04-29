// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Utilities/TMPL.hpp"

/// \brief Boundary conditions for the combined Generalized Harmonic and
/// Valencia GRMHD systems
namespace grmhd::GhValenciaDivClean::BoundaryConditions {
/// \cond
template <typename DerivedGhCondition, typename DerivedValenciaCondition>
class ProductOfConditions;
/// \endcond

namespace detail {
template <typename DerivedGhCondition, typename DerivedValenciaCondition>
using ProductOfConditionsIfConsistent = tmpl::conditional_t<
    DerivedGhCondition::bc_type == DerivedValenciaCondition::bc_type,
    ProductOfConditions<DerivedGhCondition, DerivedValenciaCondition>,
    tmpl::list<>>;

template <typename GhList, typename ValenciaList>
struct AllProductConditions;

template <typename GhList, typename... ValenciaConditions>
struct AllProductConditions<GhList, tmpl::list<ValenciaConditions...>> {
  using type = tmpl::flatten<tmpl::list<tmpl::transform<
      GhList, tmpl::bind<ProductOfConditionsIfConsistent, tmpl::_1,
                         tmpl::pin<ValenciaConditions>>>...>>;
};

template <typename ClassList>
struct remove_periodic_and_none_conditions {
  using type = tmpl::remove_if<
      ClassList,
      tmpl::or_<
          std::is_base_of<domain::BoundaryConditions::MarkAsPeriodic, tmpl::_1>,
          std::is_base_of<domain::BoundaryConditions::detail::MarkAsNone,
                          tmpl::_1>>>;
};

template <typename ClassList>
using remove_periodic_and_none_conditions_t =
    typename remove_periodic_and_none_conditions<ClassList>::type;
}  // namespace detail

/// \brief The base class for Generalized Harmonic and Valencia combined
/// boundary conditions; all boundary conditions for this system must inherit
/// from this base class.
class BoundaryCondition : public domain::BoundaryConditions::BoundaryCondition {
 public:
  // remove the periodic and none BCs from the creatable classes of the
  // individual systems; for the remaining conditions, include a
  // `ProductOfConditions` for each pair with compatible `bc_type`s.
  using creatable_classes = tmpl::push_back<
      typename detail::AllProductConditions<
          detail::remove_periodic_and_none_conditions_t<
              typename GeneralizedHarmonic::BoundaryConditions::
                  BoundaryCondition<3_st>::creatable_classes>,
          detail::remove_periodic_and_none_conditions_t<
              typename grmhd::ValenciaDivClean::BoundaryConditions::
                  BoundaryCondition::creatable_classes>>::type,
      domain::BoundaryConditions::Periodic<BoundaryCondition>>;

  BoundaryCondition() = default;
  BoundaryCondition(BoundaryCondition&&) noexcept = default;
  BoundaryCondition& operator=(BoundaryCondition&&) noexcept = default;
  BoundaryCondition(const BoundaryCondition&) = default;
  BoundaryCondition& operator=(const BoundaryCondition&) = default;
  ~BoundaryCondition() override = default;
  explicit BoundaryCondition(CkMigrateMessage* msg) noexcept;

  void pup(PUP::er& p) override;
};
}  // namespace grmhd::GhValenciaDivClean::BoundaryConditions
