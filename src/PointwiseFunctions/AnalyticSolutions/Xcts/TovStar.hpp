// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <ostream>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts::Solutions {
namespace tov_detail {

using TovCoordinates = RelativisticEuler::Solutions::TovCoordinates;

template <typename DataType>
using TovVariablesCache = cached_temp_buffer_from_typelist<tmpl::push_back<
    common_tags<DataType>,
    ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<3>, Frame::Inertial>,
    gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0>,
    gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0>,
    gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, 3>, 0>>>;

template <typename DataType>
struct TovVariables : CommonVariables<DataType, TovVariablesCache<DataType>> {
  static constexpr size_t Dim = 3;
  static constexpr int ConformalMatterScale = 0;
  using Cache = TovVariablesCache<DataType>;
  using Base = CommonVariables<DataType, TovVariablesCache<DataType>>;
  using Base::operator();

  const tnsr::I<DataType, 3>& x;
  const DataType& radius;
  const RelativisticEuler::Solutions::TovStar& tov_star;

  TovVariables(
      std::optional<std::reference_wrapper<const Mesh<Dim>>> local_mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, Dim, Frame::ElementLogical, Frame::Inertial>>>
          local_inv_jacobian,
      const tnsr::I<DataType, 3>& local_x, const DataType& local_radius,
      const RelativisticEuler::Solutions::TovStar& local_tov_star)
      : Base(std::move(local_mesh), std::move(local_inv_jacobian)),
        x(local_x),
        radius(local_radius),
        tov_star(local_tov_star) {}

  void operator()(gsl::not_null<tnsr::ii<DataType, 3>*> conformal_metric,
                  gsl::not_null<Cache*> cache,
                  Tags::ConformalMetric<DataType, 3, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::II<DataType, 3>*> inv_conformal_metric,
      gsl::not_null<Cache*> cache,
      Tags::InverseConformalMetric<DataType, 3, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::ijj<DataType, 3>*> deriv_conformal_metric,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::ii<DataType, 3>*> extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      gr::Tags::ExtrinsicCurvature<DataType, 3> /*meta*/) const override;
  void operator()(
      gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::i<DataType, 3>*> deriv_trace_extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                    tmpl::size_t<3>, Frame::Inertial> /*meta*/) const override;
  void operator()(
      gsl::not_null<Scalar<DataType>*> dt_trace_extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/)
      const override;
  void operator()(gsl::not_null<Scalar<DataType>*> conformal_factor,
                  gsl::not_null<Cache*> cache,
                  Tags::ConformalFactor<DataType> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::i<DataType, 3>*> deriv_conformal_factor,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                    Frame::Inertial> /*meta*/) const override;
  void operator()(gsl::not_null<Scalar<DataType>*> lapse,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Lapse<DataType> /*meta*/) const override;
  void operator()(gsl::not_null<tnsr::i<DataType, 3>*> deriv_lapse,
                  gsl::not_null<Cache*> cache,
                  ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<3>,
                                Frame::Inertial> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*> lapse_times_conformal_factor,
      gsl::not_null<Cache*> cache,
      Tags::LapseTimesConformalFactor<DataType> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::i<DataType, 3>*> deriv_lapse_times_conformal_factor,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<3>,
                    Frame::Inertial> /*meta*/) const override;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> shift_background,
                  gsl::not_null<Cache*> cache,
                  Tags::ShiftBackground<DataType, 3, Frame::Inertial> /*meta*/)
      const override;
  void operator()(gsl::not_null<tnsr::II<DataType, 3, Frame::Inertial>*>
                      longitudinal_shift_background_minus_dt_conformal_metric,
                  gsl::not_null<Cache*> cache,
                  Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                      DataType, 3, Frame::Inertial> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::I<DataType, 3>*> shift_excess,
      gsl::not_null<Cache*> cache,
      Tags::ShiftExcess<DataType, 3, Frame::Inertial> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::ii<DataType, 3>*> shift_strain,
      gsl::not_null<Cache*> cache,
      Tags::ShiftStrain<DataType, 3, Frame::Inertial> /*meta*/) const override;
  void operator()(gsl::not_null<Scalar<DataType>*> energy_density,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>,
                                      ConformalMatterScale> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> stress_trace,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Conformal<gr::Tags::StressTrace<DataType>,
                                      ConformalMatterScale> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> momentum_density,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, 3>,
                                      ConformalMatterScale> /*meta*/) const;

 private:
  template <typename Tag>
  typename Tag::type get_tov_var(Tag /*meta*/) const {
    // Possible optimization: Access the cache of the RelEuler::TovStar solution
    // so its intermediate quantities don't have to be re-computed repeatedly
    return get<Tag>(tov_star.variables(
        x, std::numeric_limits<double>::signaling_NaN(), tmpl::list<Tag>{}));
  }
};

}  // namespace tov_detail

/*!
 * \brief TOV solution to the XCTS equations
 *
 * \see RelativisticEuler::Solutions::TovStar
 * \see gr::Solutions::TovSolution
 */
class TovStar : public elliptic::analytic_data::AnalyticSolution {
 private:
  using RelEulerTovStar = RelativisticEuler::Solutions::TovStar;

 public:
  using options = RelEulerTovStar::options;
  static constexpr Options::String help = RelEulerTovStar::help;

  TovStar() = default;
  TovStar(const TovStar&) = default;
  TovStar& operator=(const TovStar&) = default;
  TovStar(TovStar&&) = default;
  TovStar& operator=(TovStar&&) = default;
  ~TovStar() = default;

  TovStar(double central_rest_mass_density,
          std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>
              equation_of_state,
          const RelativisticEuler::Solutions::TovCoordinates coordinate_system)
      : tov_star(central_rest_mass_density, std::move(equation_of_state),
                 coordinate_system) {}

  const EquationsOfState::EquationOfState<true, 1>& equation_of_state() const {
    return tov_star.equation_of_state();
  }

  const RelativisticEuler::Solutions::TovSolution& radial_solution() const {
    return tov_star.radial_solution();
  }

  /// \cond
  explicit TovStar(CkMigrateMessage* m)
      : elliptic::analytic_data::AnalyticSolution(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(TovStar);
  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<TovStar>(*this);
  }
  /// \endcond

  template <typename DataType>
  using tags = typename tov_detail::TovVariablesCache<DataType>::tags_list;

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables_impl<DataType>(x, std::nullopt, std::nullopt,
                                    tmpl::list<RequestedTags...>{});
  }

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x, const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables_impl<DataVector>(x, mesh, inv_jacobian,
                                      tmpl::list<RequestedTags...>{});
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    elliptic::analytic_data::AnalyticSolution::pup(p);
    p | tov_star;
  }

 private:
  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables_impl(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      std::optional<std::reference_wrapper<const Mesh<3>>> mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, 3, Frame::ElementLogical, Frame::Inertial>>>
          inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using VarsComputer = tov_detail::TovVariables<DataType>;
    typename VarsComputer::Cache cache{get_size(*x.begin())};
    const DataType radius = get(magnitude(x));
    const VarsComputer computer{std::move(mesh), std::move(inv_jacobian), x,
                                radius, tov_star};
    using unrequested_hydro_tags =
        tmpl::list_difference<hydro_tags<DataType>,
                              tmpl::list<RequestedTags...>>;
    using requested_hydro_tags =
        tmpl::list_difference<hydro_tags<DataType>, unrequested_hydro_tags>;
    tuples::tagged_tuple_from_typelist<requested_hydro_tags> hydro_vars;
    if constexpr (not std::is_same_v<requested_hydro_tags, tmpl::list<>>) {
      hydro_vars =
          tov_star.variables(x, std::numeric_limits<double>::signaling_NaN(),
                             requested_hydro_tags{});
    }
    const auto get_var = [&cache, &computer, &hydro_vars](auto tag_v) {
      using tag = std::decay_t<decltype(tag_v)>;
      if constexpr (tmpl::list_contains_v<hydro_tags<DataType>, tag>) {
        (void)cache;
        (void)computer;
        return get<tag>(hydro_vars);
      } else {
        (void)hydro_vars;
        return cache.get_var(computer, tag{});
      }
    };
    return {get_var(RequestedTags{})...};
  }

  friend bool operator==(const TovStar& lhs, const TovStar& rhs) {
    return lhs.tov_star == rhs.tov_star;
  }

  // Instead of inheriting from the RelEuler::TovStar we use an aggregate
  // pattern to avoid multiple-inheritance issues.
  RelativisticEuler::Solutions::TovStar tov_star{};
};

inline bool operator!=(const TovStar& lhs, const TovStar& rhs) {
  return not(lhs == rhs);
}

}  // namespace Xcts::Solutions
