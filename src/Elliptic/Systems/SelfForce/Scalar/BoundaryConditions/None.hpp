// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarSelfForce::BoundaryConditions {

/*!
 * \brief Applies no boundary condition at all. Used to impose nothing but
 * regularity at the horizon in horizon-penetrating coordinates or at
 * angular boundaries.
 */
class None
    : public elliptic::BoundaryConditions::BoundaryCondition<2>
      SPECTRE_FINDUS_DERIVED(None,
                             domain::BoundaryConditions::BoundaryCondition) {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<2>;

 public:
  static constexpr Options::String help =
      "Applies no boundary condition at all. Used to impose nothing but "
      "regularity at the horizon in horizon-penetrating coordinates.";
  using options = tmpl::list<>;

  None() = default;
  None(const None&) = default;
  None& operator=(const None&) = default;
  None(None&&) = default;
  None& operator=(None&&) = default;
  ~None() override = default;

  /// \cond
  WRAPPED_PUPable_decl_template(None);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override;

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return {elliptic::BoundaryConditionType::Neumann};
  }

  using argument_tags = tmpl::list<>;
  using volume_tags = tmpl::list<>;

  void apply(gsl::not_null<Scalar<ComplexDataVector>*> field,
             gsl::not_null<Scalar<ComplexDataVector>*> n_dot_field_gradient,
             const tnsr::i<ComplexDataVector, 2>& deriv_field) const;

  using argument_tags_linearized = tmpl::list<>;
  using volume_tags_linearized = tmpl::list<>;

  void apply_linearized(
      gsl::not_null<Scalar<ComplexDataVector>*> field_correction,
      gsl::not_null<Scalar<ComplexDataVector>*> n_dot_field_gradient_correction,
      const tnsr::i<ComplexDataVector, 2>& deriv_field_correction) const;

 private:
  friend bool operator==(const None& lhs, const None& rhs);
};

bool operator!=(const None& lhs, const None& rhs);

}  // namespace ScalarSelfForce::BoundaryConditions
