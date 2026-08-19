// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/TeukolskyWave.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts::AnalyticData {
namespace detail {

template <typename DataType>
using KerrSchildTeukolskyVariablesCache =
    cached_temp_buffer_from_typelist<tmpl::append<
        common_tags<DataType>,
        tmpl::list<
            gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0>,
            gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0>,
            gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, 3>, 0>,
            Tags::ConformalFactorMinusOne<DataType>,
            Tags::LapseTimesConformalFactorMinusOne<DataType>,
            Tags::ShiftExcess<DataType, 3, Frame::Inertial>>>>;

template <typename DataType>
struct KerrSchildTeukolskyVariables
    : CommonVariables<DataType, KerrSchildTeukolskyVariablesCache<DataType>> {
  static constexpr size_t Dim = 3;
  using Cache = KerrSchildTeukolskyVariablesCache<DataType>;
  using Base = CommonVariables<DataType, Cache>;
  using Base::operator();

  using kerr_schild_tags =
      tmpl::list<gr::Tags::SpatialMetric<DataType, Dim>,
                 gr::Tags::InverseSpatialMetric<DataType, Dim>,
                 gr::Tags::Lapse<DataType>, gr::Tags::Shift<DataType, Dim>,
                 gr::Tags::ExtrinsicCurvature<DataType, Dim>>;

  using teukolsky_tags = tmpl::list<
      gr::Tags::SpatialMetric<DataType, Dim, Frame::Inertial>,
      ::Tags::dt<gr::Tags::SpatialMetric<DataType, Dim, Frame::Inertial>>>;

  KerrSchildTeukolskyVariables(
      std::optional<std::reference_wrapper<const Mesh<Dim>>> local_mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, Dim, Frame::ElementLogical, Frame::Inertial>>>
          local_inv_jacobian,
      const tnsr::I<DataType, Dim>& local_x,
      const tuples::tagged_tuple_from_typelist<kerr_schild_tags>&
          local_kerr_schild_vars,
      const tuples::tagged_tuple_from_typelist<teukolsky_tags>&
          local_teukolsky_vars)
      : Base(std::move(local_mesh), std::move(local_inv_jacobian)),
        x(local_x),
        kerr_schild_vars(local_kerr_schild_vars),
        teukolsky_vars(local_teukolsky_vars) {}

  std::reference_wrapper<const tnsr::I<DataType, Dim>> x;
  std::reference_wrapper<
      const tuples::tagged_tuple_from_typelist<kerr_schild_tags>>
      kerr_schild_vars;
  std::reference_wrapper<
      const tuples::tagged_tuple_from_typelist<teukolsky_tags>>
      teukolsky_vars;

  void operator()(
      gsl::not_null<tnsr::ii<DataType, Dim>*> conformal_metric,
      gsl::not_null<Cache*> cache,
      Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_conformal_metric,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const override;
  void operator()(
      gsl::not_null<Scalar<DataType>*> dt_trace_extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/)
      const override;
  void operator()(gsl::not_null<Scalar<DataType>*> conformal_factor_minus_one,
                  gsl::not_null<Cache*> cache,
                  Xcts::Tags::ConformalFactorMinusOne<DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*> lapse_times_conformal_factor_minus_one,
      gsl::not_null<Cache*> cache,
      Xcts::Tags::LapseTimesConformalFactorMinusOne<DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> shift_background,
      gsl::not_null<Cache*> cache,
      Xcts::Tags::ShiftBackground<DataType, Dim, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::iJ<DataType, Dim>*> deriv_shift_background,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Xcts::Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
      const override;
  void operator()(gsl::not_null<tnsr::II<DataType, Dim, Frame::Inertial>*>
                      longitudinal_shift_background_minus_dt_conformal_metric,
                  gsl::not_null<Cache*> cache,
                  Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                      DataType, Dim, Frame::Inertial> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> shift_excess,
      gsl::not_null<Cache*> cache,
      Xcts::Tags::ShiftExcess<DataType, Dim, Frame::Inertial> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*> energy_density,
      gsl::not_null<Cache*> cache,
      gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*> stress_trace,
      gsl::not_null<Cache*> cache,
      gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataType, Dim>*> momentum_density,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, Dim>,
                                      0> /*meta*/) const;
};

}  // namespace detail

/*!
 * \brief Kerr-Schild plus a Teukolsky perturbation in the conformal
 * metric.
 *
 * This solution does not satisfy the constraint equations itself; it is
 * intended for supplying the free data and Dirichlet boundary conditions
 * needed to solve the XCTS equations for initial data containing a
 * Kerr-Schild black hole plus a gravitational-wave pulse.
 * In this solution, the Kerr-Schild solution supplies the lapse, shift, and
 * trace of the extrinsic curvature, while the Teukolsky wave at \f$t=0\f$ is
 * added only to the conformal spatial metric and its time derivative. The
 * spatial derivative of the conformal spatial metric is computed using
 * numerical partial derivatives.
 */
class KerrSchildTeukolsky : public elliptic::analytic_data::Background,
                            public elliptic::analytic_data::InitialGuess {
 public:
  static constexpr size_t Dim = 3;

  template <typename DataType>
  using tags =
      typename detail::KerrSchildTeukolskyVariablesCache<DataType>::tags_list;

  struct KerrSchild {
    using type = gr::Solutions::KerrSchild;
    static constexpr Options::String help{
        "Regular Cartesian Kerr-Schild black hole"};
  };
  struct TeukolskyWave {
    using type = gr::Solutions::TeukolskyWave;
    static constexpr Options::String help{
        "Teukolsky perturbation added to the conformal metric"};
  };

  using options = tmpl::list<KerrSchild, TeukolskyWave>;
  static constexpr Options::String help{
      "Kerr-Schild black hole with a Teukolsky perturbation in the XCTS "
      "conformal metric."};

  KerrSchildTeukolsky() = default;
  KerrSchildTeukolsky(const KerrSchildTeukolsky&) = default;
  KerrSchildTeukolsky& operator=(const KerrSchildTeukolsky&) = default;
  KerrSchildTeukolsky(KerrSchildTeukolsky&&) = default;
  KerrSchildTeukolsky& operator=(KerrSchildTeukolsky&&) = default;
  ~KerrSchildTeukolsky() override = default;

  KerrSchildTeukolsky(gr::Solutions::KerrSchild kerr_schild,
                      const gr::Solutions::TeukolskyWave& teukolsky_wave);

  const gr::Solutions::KerrSchild& kerr_schild() const { return kerr_schild_; }
  const gr::Solutions::TeukolskyWave& teukolsky_wave() const {
    return teukolsky_wave_;
  }

  /// \cond
  explicit KerrSchildTeukolsky(CkMigrateMessage* m)
      : elliptic::analytic_data::Background(m),
        elliptic::analytic_data::InitialGuess(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(KerrSchildTeukolsky);
  /// \endcond

  template <typename DataType, typename... RequestedTags,
            Requires<tmpl2::flat_all_v<tmpl::list_contains_v<
                tags<DataType>, RequestedTags>...>> = nullptr>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, Dim, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables_impl<DataType>(x, std::nullopt, std::nullopt,
                                    tmpl::list<RequestedTags...>{});
  }

  template <typename... RequestedTags,
            Requires<tmpl2::flat_all_v<tmpl::list_contains_v<
                tags<DataVector>, RequestedTags>...>> = nullptr>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x, const Mesh<Dim>& mesh,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables_impl<DataVector>(x, mesh, inv_jacobian,
                                      tmpl::list<RequestedTags...>{});
  }

  void pup(PUP::er& p) override;

 private:
  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables_impl(
      const tnsr::I<DataType, Dim, Frame::Inertial>& x,
      std::optional<std::reference_wrapper<const Mesh<Dim>>> mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, Dim, Frame::ElementLogical, Frame::Inertial>>>
          inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using VarsComputer = detail::KerrSchildTeukolskyVariables<DataType>;
    const auto kerr_schild_vars = kerr_schild_.variables(
        x, 0., typename VarsComputer::kerr_schild_tags{});
    const auto teukolsky_vars = teukolsky_wave_.variables(
        x, 0., typename VarsComputer::teukolsky_tags{});
    const size_t num_points = get_size(*x.begin());
    typename VarsComputer::Cache cache{num_points};
    VarsComputer computer{mesh, inv_jacobian, x, kerr_schild_vars,
                          teukolsky_vars};
    const auto get_var = [&cache, &computer](auto tag_v) {
      using tag = std::decay_t<decltype(tag_v)>;
      return cache.get_var(computer, tag{});
    };
    return {get_var(RequestedTags{})...};
  }

  friend bool operator==(const KerrSchildTeukolsky& lhs,
                         const KerrSchildTeukolsky& rhs);

  gr::Solutions::KerrSchild kerr_schild_{};
  gr::Solutions::TeukolskyWave teukolsky_wave_{};
};

bool operator==(const KerrSchildTeukolsky& lhs, const KerrSchildTeukolsky& rhs);
bool operator!=(const KerrSchildTeukolsky& lhs, const KerrSchildTeukolsky& rhs);

}  // namespace Xcts::AnalyticData
