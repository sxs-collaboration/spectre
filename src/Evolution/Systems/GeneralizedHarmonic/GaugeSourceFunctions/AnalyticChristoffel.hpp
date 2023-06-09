// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
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

  template <size_t SpatialDim>
  void gauge_and_spacetime_derivative(
      gsl::not_null<tnsr::a<DataVector, SpatialDim, Frame::Inertial>*> gauge_h,
      gsl::not_null<tnsr::ab<DataVector, SpatialDim, Frame::Inertial>*>
          d4_gauge_h,
      const Mesh<SpatialDim>& mesh, const double time,
      const tnsr::I<DataVector, SpatialDim, Frame::Inertial>& inertial_coords,
      const InverseJacobian<DataVector, SpatialDim, Frame::ElementLogical,
                            Frame::Inertial>& inverse_jacobian) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  std::unique_ptr<GaugeCondition> get_clone() const override;

 private:
  std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription_;
};
}  // namespace gh::gauges
