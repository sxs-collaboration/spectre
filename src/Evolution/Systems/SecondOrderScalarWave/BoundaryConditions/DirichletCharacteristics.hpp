// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/Tags.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"  // IWYU pragma: keep

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace Tags {
struct Time;
}  // namespace Tags
namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
/// \endcond

namespace SecondOrderScalarWave::BoundaryConditions {
/*!
 * \brief Sets boundary conditions using the characteristic decomposition: the
 * incoming mode is set from analytic data, the outgoing and zero-speed modes
 * from the interior.
 *
 * If \f$n^i\f$ is the outward pointing unit normal at the external boundary,
 * the characteristic fields and their speeds are (see
 * `SecondOrderScalarWave::characteristic_fields`):
 * - \f$v^+ = \Pi + n^i\Phi_i\f$: speed \f$+1\f$ (outgoing, taken from the
 *   interior)
 * - \f$v^- = \Pi - n^i\Phi_i\f$: speed \f$-1\f$ (incoming, taken from the
 *   analytic data, or set to zero if `AnalyticPrescription` is
 *   `ZeroIncomingMode`)
 * - \f$v^0_i\f$: speed \f$0\f$ (taken from the interior)
 *
 * The ghost \f$\Pi\f$ and \f$\Phi_i\f$ are reconstructed from these modes.
 * The ghost \f$\Psi\f$ by default is the boundary-evolved
 * `evolution::dg::Tags::BoundaryValue<Tags::Psi>`, integrated per face using
 * the interior evolution equation defined in `boundary_field_time_derivatives`.
 * If `CopyPsiFromInterior` is true the ghost \f$\Psi\f$ is instead the interior
 * evolved \f$\Psi\f$; the boundary value is still integrated but unused.
 *
 * Moving meshes are not supported: the characteristic speeds are defined
 * without a mesh velocity, so both member functions return an error if a face
 * mesh velocity is supplied.
 */
template <size_t Dim>
class DirichletCharacteristics final : public BoundaryCondition<Dim> {
 public:
  /// \brief Label for `AnalyticPrescription` setting the incoming
  /// characteristic mode \f$v^-\f$ to zero.
  struct ZeroIncomingMode {};

  /// \brief What analytic solution to prescribe.
  struct AnalyticPrescription {
    static constexpr Options::String help =
        "What analytic solution to prescribe, or ZeroIncomingMode to set the "
        "incoming characteristic mode to zero.";
    using type =
        Options::Auto<std::unique_ptr<evolution::initial_data::InitialData>,
                      ZeroIncomingMode>;
  };

  /// \brief If true, the ghost Psi is copied from the interior
  /// instead of the boundary-evolved value.
  struct CopyPsiFromInterior {
    static constexpr Options::String help =
        "If true, ghost Psi is copied from the interior instead of the "
        "boundary-evolved value.";
    using type = bool;
  };

  using options = tmpl::list<AnalyticPrescription, CopyPsiFromInterior>;

  static constexpr Options::String help{
      "Analytic boundary condition using the characteristic decomposition. The "
      "incoming mode is set from analytic data, the outgoing and zero-speed "
      "modes copied from the interior. The evolved variable Pi and the "
      "auxiliary variable Phi are reconstructed from the characteristic modes. "
      "The ghost Psi is the boundary-evolved value if CopyPsiFromInterior is "
      "false. Otherwise, the ghost Psi is copied from the interior."};

  DirichletCharacteristics() = default;
  DirichletCharacteristics(DirichletCharacteristics&&) = default;
  DirichletCharacteristics& operator=(DirichletCharacteristics&&) = default;
  DirichletCharacteristics(const DirichletCharacteristics&);
  DirichletCharacteristics& operator=(const DirichletCharacteristics&);
  ~DirichletCharacteristics() override = default;

  DirichletCharacteristics(
      std::optional<std::unique_ptr<evolution::initial_data::InitialData>>
          analytic_prescription,
      bool copy_psi_from_interior);

  explicit DirichletCharacteristics(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      domain::BoundaryConditions::BoundaryCondition, DirichletCharacteristics);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  static constexpr bool evolves_boundary_variables = true;

  using dg_interior_evolved_variables_tags =
      tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>>;
  using dg_interior_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>>;
  using dg_gridless_tags = tmpl::list<::Tags::Time>;

  using boundary_field_time_derivatives_evolved_variables_tags =
      tmpl::list<Tags::Pi, Tags::Phi<Dim>>;
  using boundary_field_time_derivatives_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>>;

  std::optional<std::string> dg_ghost(
      gsl::not_null<Scalar<DataVector>*> psi,
      gsl::not_null<Scalar<DataVector>*> pi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> phi,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const Scalar<DataVector>& boundary_psi,
      const Scalar<DataVector>& interior_psi,
      const Scalar<DataVector>& interior_pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interior_phi,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      double time) const;

  std::optional<std::string> boundary_field_time_derivatives(
      gsl::not_null<Scalar<DataVector>*> dt_boundary_psi,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const Scalar<DataVector>& boundary_psi,
      const Scalar<DataVector>& interior_pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interior_phi,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      double time) const;

 private:
  std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription_;
  bool copy_psi_from_interior_{false};
};
}  // namespace SecondOrderScalarWave::BoundaryConditions
