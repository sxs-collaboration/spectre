// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarTensor/BoundaryConditions/DirichletAnalytic.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/ScalarTensor/System.hpp"
#include "PointwiseFunctions/AnalyticData/GhScalarTensor/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ScalarTensor::BoundaryConditions {
DirichletAnalytic::DirichletAnalytic(const DirichletAnalytic& rhs)
    : PUP::able(rhs),
      BoundaryCondition{dynamic_cast<const BoundaryCondition&>(rhs)},
      analytic_prescription_(rhs.analytic_prescription_->get_clone()),
      amplitude_(rhs.amplitude_) {}

DirichletAnalytic& DirichletAnalytic::operator=(const DirichletAnalytic& rhs) {
  if (&rhs == this) {
    return *this;
  }
  analytic_prescription_ = rhs.analytic_prescription_->get_clone();
  amplitude_ = rhs.amplitude_;
  return *this;
}

DirichletAnalytic::DirichletAnalytic(
    std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription,
    const double amplitude)
    : analytic_prescription_(std::move(analytic_prescription)),
      amplitude_(amplitude) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DirichletAnalytic::get_clone() const {
  return std::make_unique<DirichletAnalytic>(*this);
}

void DirichletAnalytic::pup(PUP::er& p) {
  BoundaryCondition::pup(p);
  p | analytic_prescription_;
  p | amplitude_;
}

std::optional<std::string> DirichletAnalytic::dg_ghost(
    // GH evolved variables
    const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*>
        spacetime_metric,
    const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> pi,
    const gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*> phi,
    // Scalar evolved variables
    const gsl::not_null<Scalar<DataVector>*> psi_scalar,
    const gsl::not_null<Scalar<DataVector>*> pi_scalar,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> phi_scalar,
    // GH temporary variables
    const gsl::not_null<Scalar<DataVector>*> gamma1,
    const gsl::not_null<Scalar<DataVector>*> gamma2,
    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
    // Scalar temporary variables
    const gsl::not_null<Scalar<DataVector>*> gamma1_scalar,
    const gsl::not_null<Scalar<DataVector>*> gamma2_scalar,
    // Inverse metric
    const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
        inv_spatial_metric,
    // Mesh variables
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
    /*face_mesh_velocity*/,
    const tnsr::i<DataVector, 3, Frame::Inertial>& /*normal_covector*/,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
    // GH interior variables
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords,
    const Scalar<DataVector>& interior_gamma1,
    const Scalar<DataVector>& interior_gamma2,
    // Scalar interior variables
    const Scalar<DataVector>& gamma1_interior_scalar,
    const Scalar<DataVector>& gamma2_interior_scalar, const double time) const {
  *gamma1_scalar = gamma1_interior_scalar;
  *gamma2_scalar = gamma2_interior_scalar;
  *psi_scalar =
      make_with_value<Scalar<DataVector>>(interior_gamma1, amplitude_);
  *pi_scalar = make_with_value<Scalar<DataVector>>(*psi_scalar, 0.0);
  *phi_scalar = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      *psi_scalar, 0.0);

  *gamma1 = interior_gamma1;
  *gamma2 = interior_gamma2;
  ASSERT(analytic_prescription_ != nullptr,
         "The analytic prescription must be set.");
  using evolved_vars_tags = typename ::gh::System<3>::variables_tag::tags_list;
  auto boundary_values = call_with_dynamic_type<
      tuples::tagged_tuple_from_typelist<evolved_vars_tags>,
      gh::ScalarTensor::AnalyticData::all_analytic_data>(
      analytic_prescription_.get(),
      [&coords, &time](const auto* const analytic_solution_or_data) {
        if constexpr (is_analytic_solution_v<
                          std::decay_t<decltype(*analytic_solution_or_data)>>) {
          return analytic_solution_or_data->variables(coords, time,
                                                      evolved_vars_tags{});

        } else {
          (void)time;
          return analytic_solution_or_data->variables(coords,
                                                      evolved_vars_tags{});
        }
      });

  *spacetime_metric =
      get<gr::Tags::SpacetimeMetric<DataVector, 3>>(boundary_values);
  *pi = get<gh::Tags::Pi<DataVector, 3>>(boundary_values);
  *phi = get<gh::Tags::Phi<DataVector, 3>>(boundary_values);

  // Now compute lapse and shift...
  const auto spatial_metric = gr::spatial_metric(*spacetime_metric);
  determinant_and_inverse(lapse, inv_spatial_metric, spatial_metric);
  gr::shift(shift, *spacetime_metric, *inv_spatial_metric);
  gr::lapse(lapse, *shift, *spacetime_metric);
  return {};
}

// NOLINTNEXTLINE
PUP::able::PUP_ID DirichletAnalytic::my_PUP_ID = 0;
}  // namespace ScalarTensor::BoundaryConditions
