// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Elasticity::BoundaryConditions {
namespace detail {

template <elliptic::BoundaryConditionType BoundaryConditionType>
struct ZeroHelpString;
template <>
struct ZeroHelpString<elliptic::BoundaryConditionType::Dirichlet> {
  static constexpr Options::String help =
      "Zero Dirichlet boundary conditions imposed on the displacement vector, "
      "i.e. the elastic material is held fixed at this boundary.";
};
template <>
struct ZeroHelpString<elliptic::BoundaryConditionType::Neumann> {
  static constexpr Options::String help =
      "Zero Neumann boundary conditions imposed on the stress tensor "
      "perpendicular to the surface, i.e. the elastic material is free to "
      "deform at this boundary.";
};

template <size_t Dim, elliptic::BoundaryConditionType BoundaryConditionType>
struct ZeroImpl {
  static_assert(BoundaryConditionType ==
                        elliptic::BoundaryConditionType::Dirichlet or
                    BoundaryConditionType ==
                        elliptic::BoundaryConditionType::Neumann,
                "Unexpected boundary condition type. Supported are Dirichlet "
                "and Neumann.");

  static std::string name();
  using options = tmpl::list<>;
  static constexpr Options::String help =
      ZeroHelpString<BoundaryConditionType>::help;

  using argument_tags = tmpl::list<>;
  using volume_tags = tmpl::list<>;

  static void apply(
      gsl::not_null<tnsr::I<DataVector, Dim>*> displacement,
      gsl::not_null<tnsr::I<DataVector, Dim>*> n_dot_minus_stress);

  using argument_tags_linearized = tmpl::list<>;
  using volume_tags_linearized = tmpl::list<>;

  static void apply_linearized(
      gsl::not_null<tnsr::I<DataVector, Dim>*> displacement,
      gsl::not_null<tnsr::I<DataVector, Dim>*> n_dot_minus_stress);
};

template <size_t Dim, elliptic::BoundaryConditionType BoundaryConditionType>
bool operator==(const ZeroImpl<Dim, BoundaryConditionType>& lhs,
                const ZeroImpl<Dim, BoundaryConditionType>& rhs);

template <size_t Dim, elliptic::BoundaryConditionType BoundaryConditionType>
bool operator!=(const ZeroImpl<Dim, BoundaryConditionType>& lhs,
                const ZeroImpl<Dim, BoundaryConditionType>& rhs);

}  // namespace detail

// The following implements the registration and factory-creation mechanism

/// \cond
template <size_t Dim, elliptic::BoundaryConditionType BoundaryConditionType,
          typename Registrars>
struct Zero;

namespace Registrars {
template <size_t Dim, elliptic::BoundaryConditionType BoundaryConditionType>
struct Zero {
  template <typename Registrars>
  using f = BoundaryConditions::Zero<Dim, BoundaryConditionType, Registrars>;
};
}  // namespace Registrars
/// \endcond

/// Impose zero Dirichlet ("fixed") or Neumann ("free") boundary conditions.
template <size_t Dim, elliptic::BoundaryConditionType BoundaryConditionType,
          typename Registrars =
              tmpl::list<Registrars::Zero<Dim, BoundaryConditionType>>>
class Zero
    : public elliptic::BoundaryConditions::BoundaryCondition<Dim, Registrars>,
      public detail::ZeroImpl<Dim, BoundaryConditionType> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<Dim, Registrars>;

 public:
  Zero() = default;
  Zero(const Zero&) = default;
  Zero& operator=(const Zero&) = default;
  Zero(Zero&&) = default;
  Zero& operator=(Zero&&) = default;
  ~Zero() = default;

  /// \cond
  explicit Zero(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Zero);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<Zero>(*this);
  }
};

/// \cond
template <size_t Dim, elliptic::BoundaryConditionType BoundaryConditionType,
          typename Registrars>
PUP::able::PUP_ID Zero<Dim, BoundaryConditionType,
                       Registrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace Elasticity::BoundaryConditions
