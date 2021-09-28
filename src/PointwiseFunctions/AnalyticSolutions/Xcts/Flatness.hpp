// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <ostream>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts::Solutions {

/// \cond
template <typename Registrars>
struct Flatness;

namespace Registrars {
struct Flatness {
  template <typename Registrars>
  using f = Solutions::Flatness<Registrars>;
};
}  // namespace Registrars
/// \endcond

/// Flat spacetime in general relativity. Useful as initial guess.
template <typename Registrars = tmpl::list<Solutions::Registrars::Flatness>>
class Flatness : public AnalyticSolution<Registrars> {
 private:
  using Base = AnalyticSolution<Registrars>;

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Flat spacetime, useful as initial guess."};

  Flatness() = default;
  Flatness(const Flatness&) = default;
  Flatness& operator=(const Flatness&) = default;
  Flatness(Flatness&&) = default;
  Flatness& operator=(Flatness&&) = default;
  ~Flatness() = default;

  /// \cond
  explicit Flatness(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Flatness);
  /// \endcond

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using supported_tags_zero = tmpl::list<
        ::Tags::deriv<Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                      tmpl::size_t<3>, Frame::Inertial>,
        Tags::ConformalChristoffelFirstKind<DataType, 3, Frame::Inertial>,
        Tags::ConformalChristoffelSecondKind<DataType, 3, Frame::Inertial>,
        Tags::ConformalChristoffelContracted<DataType, 3, Frame::Inertial>,
        Tags::ConformalRicciScalar<DataVector>,
        gr::Tags::TraceExtrinsicCurvature<DataType>,
        ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>>,
        ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                      tmpl::size_t<3>, Frame::Inertial>,
        ::Tags::deriv<Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                      Frame::Inertial>,
        ::Tags::deriv<Tags::LapseTimesConformalFactor<DataType>,
                      tmpl::size_t<3>, Frame::Inertial>,
        Tags::ShiftBackground<DataType, 3, Frame::Inertial>,
        Tags::ShiftExcess<DataType, 3, Frame::Inertial>,
        Tags::ShiftStrain<DataType, 3, Frame::Inertial>,
        Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
            DataVector, 3, Frame::Inertial>,
        Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataVector>,
        Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<
            DataVector>,
        ::Tags::div<Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
            DataVector, 3, Frame::Inertial>>,
        Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataVector>,
        gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0>,
        gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0>,
        gr::Tags::Conformal<
            gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>, 0>,
        gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 6>,
        gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 6>,
        gr::Tags::Conformal<
            gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>, 6>,
        gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 8>,
        gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 8>,
        gr::Tags::Conformal<
            gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>, 8>,
        ::Tags::FixedSource<Tags::ConformalFactor<DataType>>,
        ::Tags::FixedSource<Tags::LapseTimesConformalFactor<DataType>>,
        ::Tags::FixedSource<Tags::ShiftExcess<DataType, 3, Frame::Inertial>>>;
    using supported_tags_one =
        tmpl::list<Tags::ConformalFactor<DataType>,
                   Tags::LapseTimesConformalFactor<DataType>>;
    using supported_tags_metric =
        tmpl::list<Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                   Tags::InverseConformalMetric<DataType, 3, Frame::Inertial>>;
    using supported_tags = tmpl::append<supported_tags_zero, supported_tags_one,
                                        supported_tags_metric>;
    static_assert(
        std::is_same_v<
            tmpl::list_difference<tmpl::list<RequestedTags...>, supported_tags>,
            tmpl::list<>>,
        "Not all requested tags are supported. The static_assert lists the "
        "unsupported tags.");
    const auto make_value = [&x](auto tag_v) {
      using tag = std::decay_t<decltype(tag_v)>;
      if constexpr (tmpl::list_contains_v<supported_tags_zero, tag>) {
        return make_with_value<typename tag::type>(x, 0.);
      } else if constexpr (tmpl::list_contains_v<supported_tags_one, tag>) {
        return make_with_value<typename tag::type>(x, 1.);
      } else if constexpr (tmpl::list_contains_v<supported_tags_metric, tag>) {
        auto flat_metric = make_with_value<typename tag::type>(x, 0.);
        get<0, 0>(flat_metric) = 1.;
        get<1, 1>(flat_metric) = 1.;
        get<2, 2>(flat_metric) = 1.;
        return flat_metric;
      }
    };
    return {make_value(RequestedTags{})...};
  }

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x, const Mesh<3>& /*mesh*/,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>&
      /*inv_jacobian*/,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables(x, tmpl::list<RequestedTags...>{});
  }
};

template <typename Registrars>
bool operator==(const Flatness<Registrars>& /*lhs*/,
                const Flatness<Registrars>& /*rhs*/) {
  return true;
}

template <typename Registrars>
bool operator!=(const Flatness<Registrars>& lhs,
                const Flatness<Registrars>& rhs) {
  return not(lhs == rhs);
}

/// \cond
template <typename Registrars>
PUP::able::PUP_ID Flatness<Registrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace Xcts::Solutions
