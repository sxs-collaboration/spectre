// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryConditions/BoundaryCondition.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintDampingTags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
/// \endcond

namespace ScalarTensor::BoundaryConditions {
/*!
 * \brief Sets Dirichlet boundary conditions using the analytic solution or
 * analytic data.
 */
class DirichletAnalytic final : public BoundaryCondition {
 public:
  /// \brief What analytic solution/data to prescribe.
  struct AnalyticPrescription {
    static constexpr Options::String help =
        "What analytic solution/data to prescribe.";
    using type = std::unique_ptr<evolution::initial_data::InitialData>;
  };
  struct Amplitude {
    using type = double;
    static constexpr Options::String help = {
        "Amplitude of the scalar at the boundary"};
  };

  using options = tmpl::list<AnalyticPrescription, Amplitude>;

  static constexpr Options::String help{
      "DirichletAnalytic boundary conditions."};

  DirichletAnalytic() = default;
  DirichletAnalytic(DirichletAnalytic&&) = default;
  DirichletAnalytic& operator=(DirichletAnalytic&&) = default;
  DirichletAnalytic(const DirichletAnalytic&);
  DirichletAnalytic& operator=(const DirichletAnalytic&);
  ~DirichletAnalytic() override = default;

  DirichletAnalytic(std::unique_ptr<evolution::initial_data::InitialData>
                        analytic_prescription,
                    double amplitude);

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
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>,
                 ::gh::Tags::ConstraintGamma1, ::gh::Tags::ConstraintGamma2,
                 ::CurvedScalarWave::Tags::ConstraintGamma1,
                 ::CurvedScalarWave::Tags::ConstraintGamma2>;
  using dg_gridless_tags = tmpl::list<::Tags::Time>;

  std::optional<std::string> dg_ghost(
      // GH evolved variables
      gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> spacetime_metric,
      gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> pi,
      gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*> phi,
      // Scalar evolved variables
      gsl::not_null<Scalar<DataVector>*> psi_scalar,
      gsl::not_null<Scalar<DataVector>*> pi_scalar,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> phi_scalar,
      // GH temporary variables
      gsl::not_null<Scalar<DataVector>*> gamma1,
      gsl::not_null<Scalar<DataVector>*> gamma2,
      gsl::not_null<Scalar<DataVector>*> lapse,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
      // Scalar temporary variables
      gsl::not_null<Scalar<DataVector>*> gamma1_scalar,
      gsl::not_null<Scalar<DataVector>*> gamma2_scalar,
      // Inverse metric
      gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
          inv_spatial_metric,
      // Mesh variables
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,
      // GH interior variables
      const tnsr::I<DataVector, 3, Frame::Inertial>& coords,
      const Scalar<DataVector>& interior_gamma1,
      const Scalar<DataVector>& interior_gamma2,
      // Scalar interior variables
      const Scalar<DataVector>& gamma1_interior_scalar,
      const Scalar<DataVector>& gamma2_interior_scalar, double time) const;

 private:
  std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription_;
  double amplitude_ = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace ScalarTensor::BoundaryConditions
