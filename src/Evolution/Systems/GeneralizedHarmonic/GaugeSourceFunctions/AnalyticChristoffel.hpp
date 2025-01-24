// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Mesh;
/// \endcond

namespace gh::gauges {
/*!
 * \brief Imposes the analytic gauge condition,
 * \f$H_a=\Gamma_a^{\mathrm{analytic}}\f$ from an analytic solution or analytic
 * data.
 *
 * \warning Assumes \f$\partial_t \Gamma_a=0\f$ i.e. the solution is static or
 * in harmonic gauge.
 */
class AnalyticChristoffel final : public GaugeCondition {
 private:
  template <size_t SpatialDim>
  using solution_tags =
      tmpl::list<gh::Tags::Pi<DataVector, SpatialDim>,
                 gh::Tags::Phi<DataVector, SpatialDim>,
                 gr::Tags::SpacetimeMetric<DataVector, SpatialDim>,
                 gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<DataVector, SpatialDim>,
                 gr::Tags::SpatialMetric<DataVector, SpatialDim>>;

 public:
  /// \brief What analytic solution/data to prescribe.
  struct AnalyticPrescription {
    static constexpr Options::String help =
        "What analytic solution/data to prescribe.";
    using type = std::unique_ptr<evolution::initial_data::InitialData>;
  };

  using options = tmpl::list<AnalyticPrescription>;

  static constexpr Options::String help{
      "Apply the analytic gauge condition H_a = Gamma_a, where Gamma_a comes "
      "from the AnalyticPrescription."};

  AnalyticChristoffel() = default;
  AnalyticChristoffel(const AnalyticChristoffel&);
  AnalyticChristoffel& operator=(const AnalyticChristoffel&);
  AnalyticChristoffel(AnalyticChristoffel&&) = default;
  AnalyticChristoffel& operator=(AnalyticChristoffel&&) = default;
  ~AnalyticChristoffel() override = default;

  explicit AnalyticChristoffel(
      std::unique_ptr<evolution::initial_data::InitialData>
          analytic_prescription);

  /// \cond
  explicit AnalyticChristoffel(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(AnalyticChristoffel);  // NOLINT
  /// \endcond

  template <size_t SpatialDim, class AllSolutionsForChristoffelAnalytic>
  void gauge_and_spacetime_derivative(
      const gsl::not_null<tnsr::a<DataVector, SpatialDim, Frame::Inertial>*>
          gauge_h,
      const gsl::not_null<tnsr::ab<DataVector, SpatialDim, Frame::Inertial>*>
          d4_gauge_h,
      const Mesh<SpatialDim>& mesh, const double time,
      const tnsr::I<DataVector, SpatialDim, Frame::Inertial>& inertial_coords,
      const InverseJacobian<DataVector, SpatialDim, Frame::ElementLogical,
                            Frame::Inertial>& inverse_jacobian,
      const AllSolutionsForChristoffelAnalytic /*meta*/) const {
    ASSERT(analytic_prescription_ != nullptr,
           "The analytic prescription cannot be nullptr.");
    const auto solution_vars = call_with_dynamic_type<
        tuples::tagged_tuple_from_typelist<solution_tags<SpatialDim>>,
        AllSolutionsForChristoffelAnalytic>(
        analytic_prescription_.get(),
        [&inertial_coords, &time](const auto* const analytic_solution_or_data) {
          if constexpr (is_analytic_solution_v<std::decay_t<
                            decltype(*analytic_solution_or_data)>>) {
            return analytic_solution_or_data->variables(
                inertial_coords, time, solution_tags<SpatialDim>{});

          } else {
            (void)time;
            return analytic_solution_or_data->variables(
                inertial_coords, solution_tags<SpatialDim>{});
          }
        });
    gauge_and_spacetime_derivative_impl(gauge_h, d4_gauge_h, mesh,
                                        inverse_jacobian, solution_vars);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  std::unique_ptr<GaugeCondition> get_clone() const override;

 private:
  template <size_t SpatialDim>
  void gauge_and_spacetime_derivative_impl(
      gsl::not_null<tnsr::a<DataVector, SpatialDim, Frame::Inertial>*> gauge_h,
      gsl::not_null<tnsr::ab<DataVector, SpatialDim, Frame::Inertial>*>
          d4_gauge_h,
      const Mesh<SpatialDim>& mesh,
      const InverseJacobian<DataVector, SpatialDim, Frame::ElementLogical,
                            Frame::Inertial>& inverse_jacobian,
      const tuples::tagged_tuple_from_typelist<solution_tags<SpatialDim>>&
          solution_vars) const;

  std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription_;
};
}  // namespace gh::gauges
