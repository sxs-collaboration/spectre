// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/BoundaryConditions/BoundaryCondition.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

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
 * \brief Sets Dirichlet boundary conditions from a prescribed analytic
 * solution.
 *
 * Fills the exterior (ghost) values of the evolved variables \f$\Psi\f$ and
 * \f$\Pi\f$ as well as the LDG auxiliary variable \f$\Phi_i\f$ from the
 * analytic solution evaluated at the boundary.
 */
template <size_t Dim>
class DirichletAnalytic final : public BoundaryCondition<Dim> {
 public:
  /// \brief What analytic solution to prescribe.
  struct AnalyticPrescription {
    static constexpr Options::String help =
        "What analytic solution to prescribe.";
    using type = std::unique_ptr<evolution::initial_data::InitialData>;
  };

  using options = tmpl::list<AnalyticPrescription>;

  static constexpr Options::String help{
      "DirichletAnalytic boundary conditions setting the value of Psi, Pi, and"
      " Phi to the analytic solution."};

  DirichletAnalytic() = default;
  DirichletAnalytic(DirichletAnalytic&&) = default;
  DirichletAnalytic& operator=(DirichletAnalytic&&) = default;
  DirichletAnalytic(const DirichletAnalytic&);
  DirichletAnalytic& operator=(const DirichletAnalytic&);
  ~DirichletAnalytic() override = default;

  explicit DirichletAnalytic(
      std::unique_ptr<evolution::initial_data::InitialData>
          analytic_prescription);

  explicit DirichletAnalytic(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, DirichletAnalytic);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>>;
  using dg_gridless_tags = tmpl::list<::Tags::Time>;

  std::optional<std::string> dg_ghost(
      gsl::not_null<Scalar<DataVector>*> psi,
      gsl::not_null<Scalar<DataVector>*> pi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> phi,
      const std::optional<
          tnsr::I<DataVector, Dim, Frame::Inertial>>& /*face_mesh_velocity*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*normal_covector*/,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      double time) const;

 private:
  std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription_;
};
}  // namespace SecondOrderScalarWave::BoundaryConditions
