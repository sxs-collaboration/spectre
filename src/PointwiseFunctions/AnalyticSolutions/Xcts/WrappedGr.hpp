// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Xcts::Solutions {
namespace detail {

template <typename DataType, size_t Dim>
using gr_solution_vars = tmpl::list<
    gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>,
    gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataType>,
    ::Tags::deriv<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>,
                  tmpl::size_t<Dim>, Frame::Inertial>,
    gr::Tags::Lapse<DataType>,
    ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<Dim>,
                  Frame::Inertial>,
    gr::Tags::Shift<Dim, Frame::Inertial, DataType>,
    ::Tags::deriv<gr::Tags::Shift<Dim, Frame::Inertial, DataType>,
                  tmpl::size_t<Dim>, Frame::Inertial>,
    gr::Tags::ExtrinsicCurvature<Dim, Frame::Inertial, DataType>>;

template <typename DataType>
using WrappedGrVariablesCache =
    cached_temp_buffer_from_typelist<tmpl::push_back<
        common_tags<DataType>,
        gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0>,
        gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0>,
        gr::Tags::Conformal<
            gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>, 0>>>;

template <typename DataType>
struct WrappedGrVariables
    : CommonVariables<DataType, WrappedGrVariablesCache<DataType>> {
  static constexpr size_t Dim = 3;
  using Cache = WrappedGrVariablesCache<DataType>;
  using Base = CommonVariables<DataType, WrappedGrVariablesCache<DataType>>;
  using Base::operator();

  WrappedGrVariables(
      std::optional<std::reference_wrapper<const Mesh<Dim>>> local_mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, Dim, Frame::ElementLogical, Frame::Inertial>>>
          local_inv_jacobian,
      const tnsr::I<DataType, 3>& local_x,
      const tuples::tagged_tuple_from_typelist<gr_solution_vars<DataType, Dim>>&
          local_gr_solution)
      : Base(std::move(local_mesh), std::move(local_inv_jacobian)),
        x(local_x),
        gr_solution(local_gr_solution) {}

  const tnsr::I<DataType, Dim>& x;
  const tuples::tagged_tuple_from_typelist<gr_solution_vars<DataType, Dim>>&
      gr_solution;

  void operator()(
      gsl::not_null<tnsr::ii<DataType, Dim>*> conformal_metric,
      gsl::not_null<Cache*> cache,
      Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial> /*meta*/)
      const override;
  void operator()(gsl::not_null<tnsr::II<DataType, Dim>*> inv_conformal_metric,
                  gsl::not_null<Cache*> cache,
                  Xcts::Tags::InverseConformalMetric<
                      DataType, Dim, Frame::Inertial> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_conformal_metric,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
      const override;
  void operator()(gsl::not_null<tnsr::ii<DataType, Dim>*> spatial_metric,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::SpatialMetric<Dim, Frame::Inertial,
                                          DataType> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::II<DataType, Dim>*> inv_spatial_metric,
      gsl::not_null<Cache*> cache,
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataType> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_spatial_metric,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>,
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
  void operator()(
      gsl::not_null<Scalar<DataType>*> conformal_factor,
      gsl::not_null<Cache*> cache,
      Xcts::Tags::ConformalFactor<DataType> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::i<DataType, Dim>*> conformal_factor_gradient,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<Dim>,
                    Frame::Inertial> /*meta*/) const override;
  void operator()(gsl::not_null<Scalar<DataType>*> lapse,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Lapse<DataType> /*meta*/) const override;
  void operator()(
      gsl::not_null<Scalar<DataType>*> lapse_times_conformal_factor,
      gsl::not_null<Cache*> cache,
      Xcts::Tags::LapseTimesConformalFactor<DataType> /*meta*/) const override;
  void operator()(gsl::not_null<tnsr::i<DataType, Dim>*>
                      lapse_times_conformal_factor_gradient,
                  gsl::not_null<Cache*> cache,
                  ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                                tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> shift_background,
      gsl::not_null<Cache*> cache,
      Xcts::Tags::ShiftBackground<DataType, Dim, Frame::Inertial> /*meta*/)
      const override;
  void operator()(gsl::not_null<tnsr::II<DataType, Dim, Frame::Inertial>*>
                      longitudinal_shift_background_minus_dt_conformal_metric,
                  gsl::not_null<Cache*> cache,
                  Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                      DataType, Dim, Frame::Inertial> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> shift_excess,
      gsl::not_null<Cache*> cache,
      Xcts::Tags::ShiftExcess<DataType, Dim, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::ii<DataType, Dim>*> shift_strain,
      gsl::not_null<Cache*> cache,
      Xcts::Tags::ShiftStrain<DataType, Dim, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::ii<DataType, 3>*> extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataType> /*meta*/)
      const override;
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
                  gr::Tags::Conformal<
                      gr::Tags::MomentumDensity<Dim, Frame::Inertial, DataType>,
                      0> /*meta*/) const;
};

}  // namespace detail

/*!
 * \brief XCTS quantities for a solution of the Einstein equations
 *
 * This class computes all XCTS quantities from the `GrSolution`. To do so, it
 * chooses the conformal factor
 *
 * \f{equation}{
 *   \psi = 1
 *   \text{,}
 * \f}
 *
 * so the spatial metric of the `GrSolution` is used as conformal metric,
 * \f$\bar{\gamma}_{ij = \gamma_{ij}\f$. This is particularly useful for
 * superpositions, because it means that the superposed conformal metric of two
 * `WrappedGr` solutions is probably a good conformal background to solve for a
 * binary solution (see Xcts::AnalyticData::Binary).
 *
 * For example, when the `GrSolution` is `gr::Solutions::KerrSchild`, the
 * conformal metric is the spatial Kerr metric in Kerr-Schild coordinates and
 * \f$\psi = 1\f$. It is also possible to
 * choose a different \f$\psi\f$ so the solution is non-trivial in this
 * variable, though that is probably only useful for testing and currently not
 * implemented. It should be noted, however, that the combination of
 * \f$\psi=1\f$ and apparent-horizon boundary conditions poses a hard problem to
 * the nonlinear solver when starting at a flat initial guess. This is because
 * the strongly-nonlinear boundary-conditions couple the variables in such a way
 * that the solution is initially corrected away from \f$\psi=1\f$ and is then
 * unable to recover. A conformal-factor profile such as \f$\psi=1 +
 * \frac{M}{2r}\f$ (resembling isotropic coordinates) resolves this issue. In
 * production solves this is not an issue because we choose a much better
 * initial guess than flatness, such as a superposition of Kerr solutions for
 * black-hole binary initial data.
 */
template <typename GrSolution>
class WrappedGr : public elliptic::analytic_data::AnalyticSolution,
                  public GrSolution {
 public:
  static constexpr size_t Dim = 3;

  using options = typename GrSolution::options;
  static constexpr Options::String help = GrSolution::help;
  static std::string name() { return Options::name<GrSolution>(); }

  using GrSolution::GrSolution;
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(WrappedGr<GrSolution>);
  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<WrappedGr<GrSolution>>(*this);
  }

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    const auto gr_solution =
        GrSolution::variables(x, std::numeric_limits<double>::signaling_NaN(),
                              detail::gr_solution_vars<DataType, Dim>{});
    using VarsComputer = detail::WrappedGrVariables<DataType>;
    const size_t num_points = get_size(*x.begin());
    typename VarsComputer::Cache cache{num_points};
    const VarsComputer computer{std::nullopt, std::nullopt, x, gr_solution};
    return {cache.get_var(computer, RequestedTags{})...};
  }

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x, const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const {
    const auto gr_solution =
        GrSolution::variables(x, std::numeric_limits<double>::signaling_NaN(),
                              detail::gr_solution_vars<DataType, Dim>{});
    using VarsComputer = detail::WrappedGrVariables<DataType>;
    const size_t num_points = get_size(*x.begin());
    typename VarsComputer::Cache cache{num_points};
    VarsComputer computer{mesh, inv_jacobian, x, gr_solution};
    return {cache.get_var(computer, RequestedTags{})...};
  }

  void pup(PUP::er& p) override {
    GrSolution::pup(p);
    elliptic::analytic_data::AnalyticSolution::pup(p);
  }
};

}  // namespace Xcts::Solutions
