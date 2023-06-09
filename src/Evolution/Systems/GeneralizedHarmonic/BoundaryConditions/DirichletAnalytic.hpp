// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

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
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
/// \endcond

namespace gh::BoundaryConditions {
/*!
 * \brief Sets Dirichlet boundary conditions using the analytic solution or
 * analytic data.
 */
template <size_t Dim>
class DirichletAnalytic final : public BoundaryCondition<Dim> {
 public:
  /// \brief What analytic solution/data to prescribe.
  struct AnalyticPrescription {
    static constexpr Options::String help =
        "What analytic solution/data to prescribe.";
    using type = std::unique_ptr<evolution::initial_data::InitialData>;
  };

  using options = tmpl::list<AnalyticPrescription>;

  static constexpr Options::String help{
      "DirichletAnalytic boundary conditions setting the value of the "
      "spacetime metric and its derivatives Phi and Pi to the analytic "
      "solution or analytic data."};

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
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 ::gh::ConstraintDamping::Tags::ConstraintGamma1,
                 ::gh::ConstraintDamping::Tags::ConstraintGamma2>;
  using dg_gridless_tags = tmpl::list<::Tags::Time>;

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
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      const Scalar<DataVector>& interior_gamma1,
      const Scalar<DataVector>& interior_gamma2, const double time) const;

 private:
  void lapse_shift_and_inv_spatial_metric(
      gsl::not_null<Scalar<DataVector>*> lapse,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> shift,
      gsl::not_null<tnsr::II<DataVector, Dim, Frame::Inertial>*>
          inv_spatial_metric,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric) const;

  std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription_;
};
}  // namespace gh::BoundaryConditions
