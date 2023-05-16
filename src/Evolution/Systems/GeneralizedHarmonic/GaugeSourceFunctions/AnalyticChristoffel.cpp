// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/AnalyticChristoffel.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/AllSolutions.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticData/GhGrMhd/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GhGrMhd/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GhRelativisticEuler/Factory.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace gh::gauges {
AnalyticChristoffel::AnalyticChristoffel(const AnalyticChristoffel& rhs)
    : GaugeCondition{dynamic_cast<const GaugeCondition&>(rhs)},
      analytic_prescription_(rhs.analytic_prescription_->get_clone()) {}

AnalyticChristoffel& AnalyticChristoffel::operator=(
    const AnalyticChristoffel& rhs) {
  if (&rhs == this) {
    return *this;
  }
  analytic_prescription_ = rhs.analytic_prescription_->get_clone();
  return *this;
}

AnalyticChristoffel::AnalyticChristoffel(
    std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription)
    : analytic_prescription_(std::move(analytic_prescription)) {}

AnalyticChristoffel::AnalyticChristoffel(CkMigrateMessage* const msg)
    : GaugeCondition(msg) {}

void AnalyticChristoffel::pup(PUP::er& p) {
  GaugeCondition::pup(p);
  p | analytic_prescription_;
}

std::unique_ptr<GaugeCondition> AnalyticChristoffel::get_clone() const {
  return std::make_unique<AnalyticChristoffel>(*this);
}

template <size_t SpatialDim>
void AnalyticChristoffel::gauge_and_spacetime_derivative(
    const gsl::not_null<tnsr::a<DataVector, SpatialDim, Frame::Inertial>*>
        gauge_h,
    const gsl::not_null<tnsr::ab<DataVector, SpatialDim, Frame::Inertial>*>
        d4_gauge_h,
    const Mesh<SpatialDim>& mesh, const double time,
    const tnsr::I<DataVector, SpatialDim, Frame::Inertial>& inertial_coords,
    const InverseJacobian<DataVector, SpatialDim, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian) const {
  ASSERT(analytic_prescription_ != nullptr,
         "The analytic prescription cannot be nullptr.");
  using solution_tags =
      tmpl::list<gh::Tags::Pi<DataVector, SpatialDim>,
                 gh::Tags::Phi<DataVector, SpatialDim>,
                 gr::Tags::SpacetimeMetric<DataVector, SpatialDim>,
                 gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<DataVector, SpatialDim>,
                 gr::Tags::SpatialMetric<DataVector, SpatialDim>>;
  const auto [pi, phi, spacetime_metric, lapse, shift, spatial_metric] =
      call_with_dynamic_type<tuples::tagged_tuple_from_typelist<solution_tags>,
                             solutions_including_matter<SpatialDim>>(
          analytic_prescription_.get(),
          [&inertial_coords,
           &time](const auto* const analytic_solution_or_data) {
            if constexpr (is_analytic_solution_v<std::decay_t<
                              decltype(*analytic_solution_or_data)>>) {
              return analytic_solution_or_data->variables(inertial_coords, time,
                                                          solution_tags{});

            } else {
              (void)time;
              return analytic_solution_or_data->variables(inertial_coords,
                                                          solution_tags{});
            }
          });
  // Now compute Gamma_a
  Variables<
      tmpl::list<gr::Tags::SpacetimeNormalVector<DataVector, SpatialDim>,
                 gr::Tags::SpacetimeNormalOneForm<DataVector, SpatialDim>,
                 gr::Tags::InverseSpatialMetric<DataVector, SpatialDim>,
                 gr::Tags::InverseSpacetimeMetric<DataVector, SpatialDim>>>
      temp_vars(mesh.number_of_grid_points());
  auto& spacetime_normal_vector =
      get<gr::Tags::SpacetimeNormalVector<DataVector, SpatialDim>>(temp_vars);
  auto& spacetime_normal_one_form =
      get<gr::Tags::SpacetimeNormalOneForm<DataVector, SpatialDim>>(temp_vars);
  auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, SpatialDim>>(temp_vars);
  auto& inverse_spacetime_metric =
      get<gr::Tags::InverseSpacetimeMetric<DataVector, SpatialDim>>(temp_vars);
  {
    Scalar<DataVector> det_buffer{};
    get(det_buffer)
        .set_data_ref(make_not_null(&get<0>(spacetime_normal_vector)));
    determinant_and_inverse(make_not_null(&det_buffer),
                            make_not_null(&inverse_spatial_metric),
                            spatial_metric);
    determinant_and_inverse(make_not_null(&det_buffer),
                            make_not_null(&inverse_spacetime_metric),
                            spacetime_metric);
  }
  gr::spacetime_normal_one_form(make_not_null(&spacetime_normal_one_form),
                                lapse);
  gr::spacetime_normal_vector(make_not_null(&spacetime_normal_vector), lapse,
                              shift);
  gh::trace_christoffel(gauge_h, spacetime_normal_one_form,
                        spacetime_normal_vector, inverse_spatial_metric,
                        inverse_spacetime_metric, pi, phi);
  for (auto& component : *gauge_h) {
    component *= -1.0;
  }

  tnsr::ia<DataVector, SpatialDim, Frame::Inertial> di_gauge_h{};
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      di_gauge_h.get(i, a).set_data_ref(
          make_not_null(&d4_gauge_h->get(i + 1, a)));
    }
  }
  partial_derivative(make_not_null(&di_gauge_h), *gauge_h, mesh,
                     inverse_jacobian);
  // Set time derivative to zero. We are assuming a static solution.
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    d4_gauge_h->get(0, a) = 0.0;
  }
}

// NOLINTNEXTLINE
PUP::able::PUP_ID AnalyticChristoffel::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template void AnalyticChristoffel::gauge_and_spacetime_derivative(          \
      const gsl::not_null<tnsr::a<DataVector, DIM(data), Frame::Inertial>*>   \
          gauge_h,                                                            \
      const gsl::not_null<tnsr::ab<DataVector, DIM(data), Frame::Inertial>*>  \
          d4_gauge_h,                                                         \
      const Mesh<DIM(data)>& mesh, double time,                               \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& inertial_coords, \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,     \
                            Frame::Inertial>& inverse_jacobian) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
}  // namespace gh::gauges
