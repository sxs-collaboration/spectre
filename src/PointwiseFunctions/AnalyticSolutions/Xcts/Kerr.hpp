// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Xcts::Solutions {
namespace detail {

template <typename DataType>
struct KerrVariables;

template <typename DataType>
using KerrVariablesCache = cached_temp_buffer_from_typelist<
    KerrVariables<DataType>,
    tmpl::push_back<
        common_tags<DataType>,
        Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
        ::Tags::deriv<Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                      tmpl::size_t<3>, Frame::Inertial>,
        gr::Tags::TraceExtrinsicCurvature<DataType>,
        ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>>,
        Tags::ConformalFactor<DataType>,
        ::Tags::deriv<Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                      Frame::Inertial>,
        gr::Tags::Lapse<DataType>, Tags::LapseTimesConformalFactor<DataType>,
        ::Tags::deriv<Tags::LapseTimesConformalFactor<DataType>,
                      tmpl::size_t<3>, Frame::Inertial>,
        Tags::ShiftBackground<DataType, 3, Frame::Inertial>,
        Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
            DataType, 3, Frame::Inertial>,
        Tags::ShiftExcess<DataType, 3, Frame::Inertial>,
        Tags::ShiftStrain<DataType, 3, Frame::Inertial>,
        Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0>,
        Tags::Conformal<gr::Tags::StressTrace<DataType>, 0>,
        Tags::Conformal<gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>,
                        0>>>;

template <typename DataType>
struct KerrVariables : CommonVariables<DataType, KerrVariablesCache<DataType>> {
  static constexpr size_t Dim = 3;
  using Cache = KerrVariablesCache<DataType>;
  using CommonVariables<DataType, KerrVariablesCache<DataType>>::operator();

  const tnsr::I<DataType, Dim>& x;
  mutable gr::Solutions::KerrSchild::IntermediateVars<DataType> kerr_schild;

  void operator()(
      gsl::not_null<tnsr::ii<DataType, Dim>*> conformal_metric,
      gsl::not_null<Cache*> cache,
      Tags::ConformalMetric<DataType, Dim, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<tnsr::II<DataType, Dim>*> inv_conformal_metric,
      gsl::not_null<Cache*> cache,
      Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_conformal_metric,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<Scalar<DataType>*> dt_trace_extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/)
      const noexcept;
  void operator()(gsl::not_null<Scalar<DataType>*> conformal_factor,
                  gsl::not_null<Cache*> cache,
                  Tags::ConformalFactor<DataType> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::i<DataType, Dim>*> conformal_factor_gradient,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<Dim>,
                    Frame::Inertial> /*meta*/) const noexcept;
  void operator()(gsl::not_null<Scalar<DataType>*> lapse,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Lapse<DataType> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<Scalar<DataType>*> lapse_times_conformal_factor,
      gsl::not_null<Cache*> cache,
      Tags::LapseTimesConformalFactor<DataType> /*meta*/) const noexcept;
  void operator()(gsl::not_null<tnsr::i<DataType, Dim>*>
                      lapse_times_conformal_factor_gradient,
                  gsl::not_null<Cache*> cache,
                  ::Tags::deriv<Tags::LapseTimesConformalFactor<DataType>,
                                tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> shift_background,
      gsl::not_null<Cache*> cache,
      Tags::ShiftBackground<DataType, Dim, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(gsl::not_null<tnsr::II<DataType, Dim, Frame::Inertial>*>
                      longitudinal_shift_background_minus_dt_conformal_metric,
                  gsl::not_null<Cache*> cache,
                  Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                      DataType, Dim, Frame::Inertial> /*meta*/) const noexcept;
  void operator()(gsl::not_null<tnsr::I<DataType, Dim>*> shift_excess,
                  gsl::not_null<Cache*> cache,
                  Tags::ShiftExcess<DataType, Dim, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(gsl::not_null<tnsr::ii<DataType, Dim>*> shift_strain,
                  gsl::not_null<Cache*> cache,
                  Tags::ShiftStrain<DataType, Dim, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(gsl::not_null<Scalar<DataType>*> energy_density,
                  gsl::not_null<Cache*> cache,
                  Tags::Conformal<gr::Tags::EnergyDensity<DataType>,
                                  0> /*meta*/) const noexcept;
  void operator()(gsl::not_null<Scalar<DataType>*> stress_trace,
                  gsl::not_null<Cache*> cache,
                  Tags::Conformal<gr::Tags::StressTrace<DataType>, 0> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> momentum_density,
      gsl::not_null<Cache*> cache,
      Tags::Conformal<gr::Tags::MomentumDensity<Dim, Frame::Inertial, DataType>,
                      0> /*meta*/) const noexcept;
};

}  // namespace detail

// The following implements the registration and factory-creation mechanism

/// \cond
template <typename Registrars>
struct Kerr;

namespace Registrars {
struct Kerr {
  template <typename Registrars>
  using f = Solutions::Kerr<Registrars>;
};
}  // namespace Registrars
/// \endcond

/*!
 * \brief Kerr spacetime in general relativity
 *
 * This class implements the Kerr solution to the XCTS equations. It is
 * currently implemented in Kerr-Schild coordinates only and derives most
 * quantities from the `gr::Solution::KerrSchild` class. It poses a
 * non-conformally-flat problem to the XCTS equations.
 *
 * The conformal factor in this solution is set to \f$\psi=1\f$, so the
 * conformal background-metric is the spatial Kerr metric. It is possible to
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
template <typename Registrars = tmpl::list<Solutions::Registrars::Kerr>>
class Kerr : public AnalyticSolution<Registrars>,
             public gr::Solutions::KerrSchild {
 public:
  using KerrSchild::KerrSchild;
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Kerr);

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const noexcept {
    using VarsComputer = detail::KerrVariables<DataType>;
    typename VarsComputer::Cache cache{
        get_size(*x.begin()),
        VarsComputer{
            {{std::nullopt, std::nullopt}},
            x,
            gr::Solutions::KerrSchild::IntermediateVars<DataType>{*this, x}}};
    return {cache.get_var(RequestedTags{})...};
  }

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x, const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::Logical, Frame::Inertial>&
          inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const noexcept {
    using VarsComputer = detail::KerrVariables<DataType>;
    typename VarsComputer::Cache cache{
        get_size(*x.begin()),
        VarsComputer{
            {{mesh, inv_jacobian}},
            x,
            gr::Solutions::KerrSchild::IntermediateVars<DataType>{*this, x}}};
    return {cache.get_var(RequestedTags{})...};
  }

  void pup(PUP::er& p) noexcept override {
    gr::Solutions::KerrSchild::pup(p);
    Xcts::Solutions::AnalyticSolution<Registrars>::pup(p);
  }
};

/// \cond
template <typename Registrars>
PUP::able::PUP_ID Kerr<Registrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace Xcts::Solutions
