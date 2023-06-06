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
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Flatness.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Xcts::Solutions {
namespace detail {

template <typename DataType, size_t Dim>
using gr_solution_vars =
    tmpl::list<gr::Tags::SpatialMetric<DataType, Dim>,
               gr::Tags::InverseSpatialMetric<DataType, Dim>,
               ::Tags::deriv<gr::Tags::SpatialMetric<DataType, Dim>,
                             tmpl::size_t<Dim>, Frame::Inertial>,
               gr::Tags::Lapse<DataType>,
               ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<Dim>,
                             Frame::Inertial>,
               gr::Tags::Shift<DataType, Dim>,
               ::Tags::deriv<gr::Tags::Shift<DataType, Dim>, tmpl::size_t<Dim>,
                             Frame::Inertial>,
               gr::Tags::ExtrinsicCurvature<DataType, Dim>>;

template <typename DataType>
using WrappedGrVariablesCache =
    cached_temp_buffer_from_typelist<tmpl::push_back<
        common_tags<DataType>,
        hydro::Tags::MagneticFieldDotSpatialVelocity<DataType>,
        hydro::Tags::ComovingMagneticFieldSquared<DataType>,
        gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0>,
        gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0>,
        gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, 3>, 0>>>;

template <typename DataType, bool HasMhd>
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
          local_gr_solution,
      const tuples::tagged_tuple_from_typelist<hydro_tags<DataType>>&
          local_hydro_solution)
      : Base(std::move(local_mesh), std::move(local_inv_jacobian)),
        x(local_x),
        gr_solution(local_gr_solution),
        hydro_solution(local_hydro_solution) {}

  const tnsr::I<DataType, Dim>& x;
  const tuples::tagged_tuple_from_typelist<gr_solution_vars<DataType, Dim>>&
      gr_solution;
  const tuples::tagged_tuple_from_typelist<hydro_tags<DataType>>&
      hydro_solution;

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
  void operator()(
      gsl::not_null<tnsr::ii<DataType, Dim>*> spatial_metric,
      gsl::not_null<Cache*> cache,
      gr::Tags::SpatialMetric<DataType, Dim> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::II<DataType, Dim>*> inv_spatial_metric,
      gsl::not_null<Cache*> cache,
      gr::Tags::InverseSpatialMetric<DataType, Dim> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_spatial_metric,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<gr::Tags::SpatialMetric<DataType, Dim>, tmpl::size_t<Dim>,
                    Frame::Inertial> /*meta*/) const override;
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
      gr::Tags::ExtrinsicCurvature<DataType, 3> /*meta*/) const override;
  void operator()(
      gsl::not_null<Scalar<DataType>*> magnetic_field_dot_spatial_velocity,
      gsl::not_null<Cache*> cache,
      hydro::Tags::MagneticFieldDotSpatialVelocity<DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*> comoving_magnetic_field_squared,
      gsl::not_null<Cache*> cache,
      hydro::Tags::ComovingMagneticFieldSquared<DataType> /*meta*/) const;
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
 *
 * \warning
 * The computation of the XCTS matter source terms (energy density $\rho$,
 * momentum density $S^i$, stress trace $S$) uses GR quantities (lapse $\alpha$,
 * shift $\beta^i$, spatial metric $\gamma_{ij}$), which means these GR
 * quantities are not treated dynamically in the source terms when solving the
 * XCTS equations. If the GR quantities satisfy the Einstein constraints (as is
 * the case if the `GrSolution` is actually a solution to the Einstein
 * equations), then the XCTS solve will reproduce the GR quantities given the
 * fixed sources computed here. However, if the GR quantities don't satisfy the
 * Einstein constraints (e.g. because a magnetic field was added to the solution
 * but ignored in the gravity sector, or because it is a hydrodynamic solution
 * on a fixed background metric) then the XCTS solution will depend on our
 * treatment of the source terms: fixing the source terms (the simple approach
 * taken here) means we're making a choice of $W$ and $u^i$. This is what
 * initial data codes usually do when they iterate back and forth between a
 * hydro solve and an XCTS solve (e.g. see \cite Tacik2016zal). Alternatively,
 * we could fix $v^i$ and compute $W$ and $u^i$ from $v^i$ and the dynamic
 * metric variables at every step in the XCTS solver algorithm. This requires
 * adding the source terms and their linearization to the XCTS equations, and
 * could be interesting to explore.
 *
 * \tparam GrSolution Any solution to the Einstein constraint equations
 * \tparam HasMhd Enable to compute matter source terms. Disable to set matter
 * source terms to zero.
 */
template <typename GrSolution, bool HasMhd = false,
          typename GrSolutionOptions = typename GrSolution::options>
class WrappedGr;

template <typename GrSolution, bool HasMhd, typename... GrSolutionOptions>
class WrappedGr<GrSolution, HasMhd, tmpl::list<GrSolutionOptions...>>
    : public elliptic::analytic_data::AnalyticSolution {
 public:
  static constexpr size_t Dim = 3;

  using options = typename GrSolution::options;
  static constexpr Options::String help = GrSolution::help;
  static std::string name() { return pretty_type::name<GrSolution>(); }

  WrappedGr() = default;
  WrappedGr(const WrappedGr&) = default;
  WrappedGr& operator=(const WrappedGr&) = default;
  WrappedGr(WrappedGr&&) = default;
  WrappedGr& operator=(WrappedGr&&) = default;
  ~WrappedGr() = default;

  WrappedGr(typename GrSolutionOptions::type... gr_solution_options)
      : gr_solution_(std::move(gr_solution_options)...) {}

  const GrSolution& gr_solution() const { return gr_solution_; }

  /// \cond
  explicit WrappedGr(CkMigrateMessage* m)
      : elliptic::analytic_data::AnalyticSolution(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(WrappedGr);
  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<WrappedGr>(*this);
  }
  /// \endcond

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables_impl<DataType>(x, std::nullopt, std::nullopt,
                                    tmpl::list<RequestedTags...>{});
  }

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x, const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables_impl<DataVector>(x, mesh, inv_jacobian,
                                      tmpl::list<RequestedTags...>{});
  }

  void pup(PUP::er& p) override {
    elliptic::analytic_data::AnalyticSolution::pup(p);
    p | gr_solution_;
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
    tuples::tagged_tuple_from_typelist<detail::gr_solution_vars<DataType, Dim>>
        gr_solution;
    if constexpr (is_analytic_solution_v<GrSolution>) {
      gr_solution = gr_solution_.variables(
          x, std::numeric_limits<double>::signaling_NaN(),
          detail::gr_solution_vars<DataType, Dim>{});
    } else {
      gr_solution =
          gr_solution_.variables(x, detail::gr_solution_vars<DataType, Dim>{});
    }
    tuples::tagged_tuple_from_typelist<hydro_tags<DataType>> hydro_solution;
    if constexpr (HasMhd) {
      if constexpr (is_analytic_solution_v<GrSolution>) {
        hydro_solution = gr_solution_.variables(
            x, std::numeric_limits<double>::signaling_NaN(),
            hydro_tags<DataType>{});
      } else {
        hydro_solution = gr_solution_.variables(x, hydro_tags<DataType>{});
      }
    }
    using VarsComputer = detail::WrappedGrVariables<DataType, HasMhd>;
    const size_t num_points = get_size(*x.begin());
    typename VarsComputer::Cache cache{num_points};
    VarsComputer computer{mesh, inv_jacobian, x, gr_solution, hydro_solution};
    const auto get_var = [&cache, &computer, &hydro_solution, &x](auto tag_v) {
      using tag = std::decay_t<decltype(tag_v)>;
      if constexpr (tmpl::list_contains_v<hydro_tags<DataType>, tag>) {
        (void)cache;
        (void)computer;
        if constexpr (HasMhd) {
          (void)x;
          return get<tag>(hydro_solution);
        } else {
          (void)hydro_solution;
          return get<tag>(Flatness{}.variables(x, tmpl::list<tag>{}));
        }
      } else {
        (void)hydro_solution;
        (void)x;
        return cache.get_var(computer, tag{});
      }
    };
    return {get_var(RequestedTags{})...};
  }

  friend bool operator==(const WrappedGr<GrSolution, HasMhd>& lhs,
                         const WrappedGr<GrSolution, HasMhd>& rhs) {
    return lhs.gr_solution_ == rhs.gr_solution_;
  }

  GrSolution gr_solution_;
};

template <typename GrSolution, bool HasMhd>
inline bool operator!=(const WrappedGr<GrSolution, HasMhd>& lhs,
                       const WrappedGr<GrSolution, HasMhd>& rhs) {
  return not(lhs == rhs);
}

template <typename GrMhdSolution>
using WrappedGrMhd = WrappedGr<GrMhdSolution, true>;

}  // namespace Xcts::Solutions
