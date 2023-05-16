// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DirichletAnalytic.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>
#include <type_traits>

#include "Evolution/Systems/GeneralizedHarmonic/AllSolutions.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace gh::BoundaryConditions {
template <size_t Dim>
DirichletAnalytic<Dim>::DirichletAnalytic(const DirichletAnalytic& rhs)
    : BoundaryCondition<Dim>{dynamic_cast<const BoundaryCondition<Dim>&>(rhs)},
      analytic_prescription_(rhs.analytic_prescription_->get_clone()) {}

template <size_t Dim>
DirichletAnalytic<Dim>& DirichletAnalytic<Dim>::operator=(
    const DirichletAnalytic& rhs) {
  if (&rhs == this) {
    return *this;
  }
  analytic_prescription_ = rhs.analytic_prescription_->get_clone();
  return *this;
}

template <size_t Dim>
DirichletAnalytic<Dim>::DirichletAnalytic(
    std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription)
    : analytic_prescription_(std::move(analytic_prescription)) {}

template <size_t Dim>
DirichletAnalytic<Dim>::DirichletAnalytic(CkMigrateMessage* const msg)
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DirichletAnalytic<Dim>::get_clone() const {
  return std::make_unique<DirichletAnalytic>(*this);
}

template <size_t Dim>
void DirichletAnalytic<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
  p | analytic_prescription_;
}

template <size_t Dim>
std::optional<std::string> DirichletAnalytic<Dim>::dg_ghost(
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
    const Scalar<DataVector>& interior_gamma2, const double time) const {
  *gamma1 = interior_gamma1;
  *gamma2 = interior_gamma2;
  ASSERT(analytic_prescription_ != nullptr,
         "The analytic prescription must be set.");
  using evolved_vars_tags = typename System<Dim>::variables_tag::tags_list;
  auto boundary_values = call_with_dynamic_type<
      tuples::tagged_tuple_from_typelist<evolved_vars_tags>,
      solutions_including_matter<Dim>>(
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
      get<gr::Tags::SpacetimeMetric<DataVector, Dim>>(boundary_values);
  *pi = get<gh::Tags::Pi<DataVector, Dim>>(boundary_values);
  *phi = get<gh::Tags::Phi<DataVector, Dim>>(boundary_values);

  // Now compute lapse and shift...
  lapse_shift_and_inv_spatial_metric(lapse, shift, inv_spatial_metric,
                                     *spacetime_metric);
  return {};
}

template <size_t Dim>
void DirichletAnalytic<Dim>::lapse_shift_and_inv_spatial_metric(
    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> shift,
    const gsl::not_null<tnsr::II<DataVector, Dim, Frame::Inertial>*>
        inv_spatial_metric,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric) const {
  const auto spatial_metric = gr::spatial_metric(spacetime_metric);
  determinant_and_inverse(lapse, inv_spatial_metric, spatial_metric);
  gr::shift(shift, spacetime_metric, *inv_spatial_metric);
  gr::lapse(lapse, *shift, spacetime_metric);
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID DirichletAnalytic<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template class DirichletAnalytic<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace gh::BoundaryConditions
