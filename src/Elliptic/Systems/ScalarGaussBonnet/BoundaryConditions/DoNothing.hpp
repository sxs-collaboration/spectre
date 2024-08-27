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

namespace sgb::BoundaryConditions {

/// Do not apply a boundary condition, used exclusively for singular boundary
/// value problems.

class DoNothing : public elliptic::BoundaryConditions::BoundaryCondition<3> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<3>;

 public:
  static constexpr Options::String help =
      "Do not apply a boundary condition, used exclusively for singular "
      "boundary value problems.";

  using options = tmpl::list<>;

  DoNothing() = default;
  DoNothing(const DoNothing&) = default;
  DoNothing& operator=(const DoNothing&) = default;
  DoNothing(DoNothing&&) = default;
  DoNothing& operator=(DoNothing&&) = default;
  ~DoNothing() override = default;

  /// \cond
  explicit DoNothing(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(DoNothing);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<DoNothing>(*this);
  }

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return {1, elliptic::BoundaryConditionType::Dirichlet};
  }

  using argument_tags = tmpl::list<>;
  using volume_tags = tmpl::list<>;

  void apply(gsl::not_null<Scalar<DataVector>*> /*field*/,
             gsl::not_null<Scalar<DataVector>*> /*n_dot_field_gradient*/,
             const tnsr::i<DataVector, 3>& /*deriv_field*/) const {};

  using argument_tags_linearized = tmpl::list<>;
  using volume_tags_linearized = tmpl::list<>;

  void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> /*field_correction*/,
      gsl::not_null<Scalar<DataVector>*>
      /*n_dot_field_gradient_correction*/,
      const tnsr::i<DataVector, 3>& /*deriv_field_correction*/) const {};
};

inline bool operator==(const DoNothing& /*lhs*/, const DoNothing& /*rhs*/) {
  return true;
}

inline bool operator!=(const DoNothing& /*lhs*/, const DoNothing& /*rhs*/) {
  return false;
}

/// \cond
PUP::able::PUP_ID DoNothing::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace sgb::BoundaryConditions
