// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
/// \endcond

namespace GeneralizedHarmonic::BoundaryConditions {
/*!
 * \brief Sets Dirichlet boundary conditions to a Minkowski spacetime.
 */
template <size_t Dim>
class DirichletMinkowski final : public BoundaryCondition<Dim> {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "DirichletMinkowski boundary conditions setting the value of the "
      "spacetime metric and its derivatives Phi and Pi to Minkowski (i.e., "
      "flat spacetime)."};

  DirichletMinkowski() = default;
  DirichletMinkowski(DirichletMinkowski&&) = default;
  DirichletMinkowski& operator=(DirichletMinkowski&&) = default;
  DirichletMinkowski(const DirichletMinkowski&) = default;
  DirichletMinkowski& operator=(const DirichletMinkowski&) = default;
  ~DirichletMinkowski() override = default;

  explicit DirichletMinkowski(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, DirichletMinkowski);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags = tmpl::list<
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1,
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2>;
  using dg_gridless_tags = tmpl::list<>;

  std::optional<std::string> dg_ghost(
      const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          spacetime_metric,
      const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*> pi,
      const gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*> phi,
      const gsl::not_null<Scalar<DataVector>*> gamma1,
      const gsl::not_null<Scalar<DataVector>*> gamma2,
      const gsl::not_null<Scalar<DataVector>*> lapse,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> shift,
      const gsl::not_null<tnsr::II<DataVector, Dim, Frame::Inertial>*>
          inv_spatial_metric,
      const std::optional<
          tnsr::I<DataVector, Dim, Frame::Inertial>>& /*face_mesh_velocity*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*normal_covector*/,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& /*normal_vector*/,
      const Scalar<DataVector>& interior_gamma1,
      const Scalar<DataVector>& interior_gamma2) const;
};
}  // namespace GeneralizedHarmonic::BoundaryConditions
