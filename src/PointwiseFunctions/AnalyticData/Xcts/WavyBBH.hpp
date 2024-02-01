// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts::AnalyticData {

namespace detail {

template <typename DataType>
using WavyBBHVariablesCache = cached_temp_buffer_from_typelist<tmpl::append<
  common_tags<DataType>,
  tmpl::list<
      ::Tags::deriv<Tags::ShiftBackground<DataType, 3, Frame::Inertial>,
              tmpl::size_t<3>, Frame::Inertial>,
      gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0>,
      gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0>,
      gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, 3>, 0>,
      // For initial guesses
      Tags::ConformalFactorMinusOne<DataType>,
      Tags::LapseTimesConformalFactorMinusOne<DataType>,
      Tags::ShiftExcess<DataType, 3, Frame::Inertial>>,
    hydro_tags<DataType>>>;

template <typename DataType>
struct WavyBBHVariables
    : CommonVariables<DataType, WavyBBHVariablesCache<DataType>> {
  static constexpr size_t Dim = 3;
  using Cache = WavyBBHVariablesCache<DataType>;
  using Base = CommonVariables<DataType, WavyBBHVariablesCache<DataType>>;
  using Base::operator();

  WavyBBHVariables(
      std::optional<std::reference_wrapper<const Mesh<Dim>>> local_mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, Dim, Frame::ElementLogical, Frame::Inertial>>>
          local_inv_jacobian,
      const tnsr::I<DataType, 3>& local_x)
      : Base(std::move(local_mesh), std::move(local_inv_jacobian)),
        x(local_x) {}

  const tnsr::I<DataType, 3>& x;

  void operator()(
      const gsl::not_null<tnsr::ii<DataType, Dim>*> conformal_metric,
      const gsl::not_null<Cache*> cache,
      Tags::ConformalMetric<DataType, Dim, Frame::Inertial> meta)
      const override;
  void operator()(
      const gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_conformal_metric,
      const gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial>
          meta) const override;
  void operator()(
      const gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
      const gsl::not_null<Cache*> cache,
      gr::Tags::TraceExtrinsicCurvature<DataType> meta) const override;
  void operator()(
      const gsl::not_null<Scalar<DataType>*> dt_trace_extrinsic_curvature ,
      const gsl::not_null<Cache*> cache,
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> meta)
      const override;
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> shift_background,
      gsl::not_null<Cache*> cache,
      Tags::ShiftBackground<DataType, Dim, Frame::Inertial> /*meta*/)
      const override;
  void operator()(gsl::not_null<tnsr::II<DataType, Dim, Frame::Inertial>*>
                      longitudinal_shift_background_minus_dt_conformal_metric,
                  gsl::not_null<Cache*> cache,
                  Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                      DataType, Dim, Frame::Inertial> /*meta*/) const override;
};

}  // namespace detail

/*!
 * \brief   WavyBBH black hole initial data with realistic wave background,
 * constructed in Post-Newtonian approximation.
 *
 * This class implements background data for the XCTS equations describing...
 */
class WavyBBH : public elliptic::analytic_data::Background,
               public elliptic::analytic_data::InitialGuess {
 public:

  //ADD NECESSARY MORE OPTION (see WavyBBH.hpp as example) --- JR

  struct MassLeft {
    static constexpr Options::String help =
        "BLA BLA BLA";
    using type = double;
  };
  struct MassRight {
    static constexpr Options::String help =
        "BLA BLA BLA";
    using type = double;
  };
  using options = tmpl::list<MassLeft, MassRight>;
  static constexpr Options::String help =
      "BLA BLA BLA";

  WavyBBH() = default;
  WavyBBH(const WavyBBH&) = delete;
  WavyBBH& operator=(const WavyBBH&) = delete;
  WavyBBH(WavyBBH&&) = default;
  WavyBBH& operator=(WavyBBH&&) = default;
  ~WavyBBH() = default;

  WavyBBH(double mass_left, double mass_right,
         const Options::Context& context = {})
      : mass_left_(mass_left), mass_right_(mass_right) {
    if (mass_left_ >= 0 or mass_right_ >= 0) {
      PARSE_ERROR(context, "'MassLeft' and 'MassRight' need to be positive.");
    }
  }

  /// \cond
  explicit WavyBBH(CkMigrateMessage* m)
      : elliptic::analytic_data::Background(m),
      elliptic::analytic_data::InitialGuess(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(WavyBBH);
  /// \endcond

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

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override {
    elliptic::analytic_data::Background::pup(p);
    elliptic::analytic_data::InitialGuess::pup(p);
    p | mass_left_;
    p | mass_right_;
  }

  double mass_left() const { return mass_left_; }
  double mass_right() const { return mass_right_; }

 private:
  double mass_left_ = std::numeric_limits<double>::signaling_NaN();
  double mass_right_ = std::numeric_limits<double>::signaling_NaN();

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables_impl(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      std::optional<std::reference_wrapper<const Mesh<3>>> mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, 3, Frame::ElementLogical, Frame::Inertial>>>
          inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using VarsComputer = detail::WavyBBHVariables<DataType>;
    typename VarsComputer::Cache cache{get_size(*x.begin())};
    const VarsComputer computer{std::move(mesh), std::move(inv_jacobian), x};

    //SEE IF THERE IS ANYTHING TO ADD HERE --- JR

    return {get_var(RequestedTags{})...};
  }
};

}  // namespace Xcts::AnalyticData
