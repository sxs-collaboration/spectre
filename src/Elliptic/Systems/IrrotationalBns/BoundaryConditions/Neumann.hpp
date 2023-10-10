// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace IrrotationalBns ::BoundaryConditions {

/// Impose Neumann boundary conditions \f$a u + b n_i \nabla^i u = c\f$. The
/// boundary condition is imposed as Neumann-type (i.e. on \f$n_i \nabla^i u\f$)
/// if \f$|b| > 0\f$ and as Dirichlet-type (i.e. on \f$u\f$) if \f$b = 0\f$.
class Neumann : public elliptic::BoundaryConditions::BoundaryCondition<3> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<3>;

 public:
  static constexpr Options::String help =
      "Neumann boundary conditions  n^i U_i = 0.";

  using options = tmpl::list<>;

  Neumann() = default;
  Neumann(const Neumann&) = default;
  Neumann& operator=(const Neumann&) = default;
  Neumann(Neumann&&) = default;
  Neumann& operator=(Neumann&&) = default;
  ~Neumann() = default;

  /// \cond
  explicit Neumann(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Neumann);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<Neumann>(*this);
  }

  Neumann(const Options::Context& context = {});

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return {elliptic::BoundaryConditionType::Neumann};
  }

  using argument_tags = tmpl::list<>;
  using volume_tags = tmpl::list<>;

  void apply(gsl::not_null<Scalar<DataVector>*> velocity_potential,
             gsl::not_null<Scalar<DataVector>*> n_dot_auxiliary_velocity) const;

  using argument_tags_linearized = tmpl::list<>;
  using volume_tags_linearized = tmpl::list<>;

  void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> velocity_potential_correction,
      gsl::not_null<Scalar<DataVector>*> n_dot_auxiliary_velocity_correction)
      const;

  void pup(PUP::er& p) override;

 private:
};

template <size_t Dim>
bool operator==(const Neumann<Dim>& lhs, const Neumann<Dim>& rhs);

template <size_t Dim>
bool operator!=(const Neumann<Dim>& lhs, const Neumann<Dim>& rhs);

}  // namespace IrrotationalBns::BoundaryConditions
