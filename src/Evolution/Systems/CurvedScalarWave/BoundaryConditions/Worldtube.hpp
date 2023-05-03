// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave::BoundaryConditions {

/*!
 * \brief Sets boundary conditions for the elements abutting the worldtube using
 * a combination of constraint-preserving boundary conditions and the local
 * solution evolved inside the worldtube.
 *
 * \details After extensive experimentation we found this set of boundary
 * conditions to be optimal. They are formulated in terms of characteristic
 * fields.
 *
 * Boundary conditions for \f$w^0_\psi\f$ \f$w^0_i\f$ are formulated
 * by demanding that there are no constraint violations flowing into the
 * numerical domain and are formulated as corrections to the time derivative of
 * the evolved variables directly. The derivation is described in
 * \ref ConstraintPreservingSphericalRadiation .
 *
 * Boundary conditions for \f$\w^-\f$ are formulated by evaluating the
 * analytical solution of the worldtube at the grid points of each element
 * abutting the worldtube. The data is updated and saved to
 * \ref CurvedScalarWave::Worldtube::Tags::WorldtubeSolution each time step.
 * It is then treated like a ghost field and applied with the chosen numerical
 * flux. So far only the upwind flux has been tried.
 *
 * We tried several other combinations such as setting all fields from the
 * worldtube solution but found that this caused major constraint violations to
 * flow out of the worldtube.
 *
 * \note We found that, depending on the worldtube size, a fairly high value for
 * \f$\gamma_2\f$ of ca. 10 is required near the worldtube to ensure a stable
 * evolution when using these boundary conditions.
 */
template <size_t Dim>
class Worldtube final : public BoundaryConditions::BoundaryCondition<Dim> {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Boundary conditions set by the worldtube. w^- will be set by the "
      "internal worldtube solution, w^psi and w^0_i are fixed by constraint "
      "preserving boundary conditions on the time derivative."};

  Worldtube() = default;
  explicit Worldtube(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, Worldtube);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::GhostAndTimeDerivative;

  void pup(PUP::er& p) override;

  using dg_interior_temporary_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, Dim>,
                 gr::Tags::InverseSpatialMetric<DataVector, Dim>,
                 Tags::ConstraintGamma1, Tags::ConstraintGamma2>;

  using dg_gridless_tags =
      tmpl::list<CurvedScalarWave::Worldtube::Tags::WorldtubeSolution<Dim>>;

  using dg_interior_evolved_variables_tags =
      tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>>;
  using dg_interior_dt_vars_tags = tmpl::list<::Tags::dt<Tags::Psi>>;
  using dg_interior_deriv_vars_tags = tmpl::list<
      ::Tags::deriv<Tags::Psi, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::deriv<Tags::Phi<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>;

  std::optional<std::string> dg_time_derivative(
      gsl::not_null<Scalar<DataVector>*> dt_psi_correction,
      gsl::not_null<Scalar<DataVector>*> dt_pi_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          dt_phi_correction,
      const std::optional<tnsr::I<DataVector, Dim>>& face_mesh_velocity,
      const tnsr::i<DataVector, Dim>& normal_covector,
      const tnsr::I<DataVector, Dim>& normal_vector,
      const Scalar<DataVector>& /*psi*/, const Scalar<DataVector>& /*pi*/,
      const tnsr::i<DataVector, Dim>& phi, const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim>& shift,
      const tnsr::II<DataVector, Dim,
                     Frame::Inertial>& /*inverse_spatial_metric*/,
      const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
      const Scalar<DataVector>& /*dt_psi*/,
      const tnsr::i<DataVector, Dim>& d_psi,
      const tnsr::ij<DataVector, Dim>& d_phi,
      const Variables<tmpl::list<Tags::Psi, Tags::Pi,
                                 Tags::Phi<Dim>>>& /*worldtube_vars*/) const;

  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> psi,
      const gsl::not_null<Scalar<DataVector>*> pi,
      const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> phi,
      const gsl::not_null<Scalar<DataVector>*> lapse,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> shift,
      const gsl::not_null<Scalar<DataVector>*> gamma1,
      const gsl::not_null<Scalar<DataVector>*> gamma2,
      const gsl::not_null<tnsr::II<DataVector, Dim, Frame::Inertial>*>
          inverse_spatial_metric,

      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*face_mesh_velocity*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
      const Scalar<DataVector>& psi_interior,
      const Scalar<DataVector>& pi_interior,
      const tnsr::i<DataVector, Dim>& phi_interior,
      const Scalar<DataVector>& lapse_interior,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_interior,
      const tnsr::II<DataVector, Dim, Frame::Inertial>&
          inverse_spatial_metric_interior,
      const Scalar<DataVector>& gamma1_interior,
      const Scalar<DataVector>& gamma2_interior,
      const Scalar<DataVector>& /*dt_psi*/,
      const tnsr::i<DataVector, Dim>& d_psi,
      const tnsr::ij<DataVector, Dim>& /*d_phi*/,
      const Variables<tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>>>&
          worldtube_vars) const;
};
}  // namespace CurvedScalarWave::BoundaryConditions
