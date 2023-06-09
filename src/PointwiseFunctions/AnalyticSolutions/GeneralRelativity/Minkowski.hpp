// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"  // for tags
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace Tags {
template <typename Tag>
struct dt;
}  // namespace Tags
/// \endcond

namespace gr {
namespace Solutions {

/*!
 * \brief The Minkowski solution for flat space in Dim spatial dimensions.
 *
 * \details The solution has lapse \f$N(x,t)= 1 \f$, shift \f$N^i(x,t) = 0 \f$
 * and the identity as the spatial metric: \f$g_{ii} = 1 \f$
 */
template <size_t Dim>
class Minkowski : public AnalyticSolution<Dim>, public MarkAsAnalyticSolution {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Minkowski solution to Einstein's Equations"};


  Minkowski() = default;
  Minkowski(const Minkowski& /*rhs*/) = default;
  Minkowski& operator=(const Minkowski& /*rhs*/) = default;
  Minkowski(Minkowski&& /*rhs*/) = default;
  Minkowski& operator=(Minkowski&& /*rhs*/) = default;
  ~Minkowski() = default;

  explicit Minkowski(CkMigrateMessage* /*msg*/);

  template <typename DataType>
  using DerivLapse = ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<Dim>,
                                   Frame::Inertial>;
  template <typename DataType>
  using DerivShift = ::Tags::deriv<gr::Tags::Shift<DataType, Dim>,
                                   tmpl::size_t<Dim>, Frame::Inertial>;
  template <typename DataType>
  using DerivSpatialMetric =
      ::Tags::deriv<gr::Tags::SpatialMetric<DataType, Dim>, tmpl::size_t<Dim>,
                    Frame::Inertial>;

  template <typename DataType, typename Frame = Frame::Inertial>
  using tags = tmpl::flatten<tmpl::list<
      typename AnalyticSolution<Dim>::template tags<DataType, Frame>,
      gr::Tags::DerivDetSpatialMetric<DataType, Dim, Frame>,
      gr::Tags::TraceExtrinsicCurvature<DataType>,
      gr::Tags::SpatialChristoffelFirstKind<DataType, Dim, Frame>,
      gr::Tags::SpatialChristoffelSecondKind<DataType, Dim, Frame>,
      gr::Tags::TraceSpatialChristoffelSecondKind<DataType, Dim, Frame>>>;

  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, Dim>& x,
                                         double t,
                                         tmpl::list<Tags...> /*meta*/) const {
    static_assert(
        tmpl2::flat_all_v<
            tmpl::list_contains_v<tags<DataType>, Tags>...>,
        "At least one of the requested tags is not supported. The requested "
        "tags are listed as template parameters of the `variables` function.");

    return {get<Tags>(variables(x, t, tmpl::list<Tags>{}))...};
  }

  template <typename DataType>
  tuples::TaggedTuple<gr::Tags::Lapse<DataType>> variables(
      const tnsr::I<DataType, Dim>& x, double t,
      tmpl::list<gr::Tags::Lapse<DataType>> /*meta*/) const;

  template <typename DataType>
  tuples::TaggedTuple<::Tags::dt<gr::Tags::Lapse<DataType>>> variables(
      const tnsr::I<DataType, Dim>& x, double t,
      tmpl::list<::Tags::dt<gr::Tags::Lapse<DataType>>> /*meta*/) const;

  template <typename DataType>
  tuples::TaggedTuple<::Tags::deriv<gr::Tags::Lapse<DataType>,
                                    tmpl::size_t<Dim>, Frame::Inertial>>
  variables(
      const tnsr::I<DataType, Dim>& x, double t,
      tmpl::list<::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<Dim>,
                               Frame::Inertial>> /*meta*/) const;

  template <typename DataType>
  tuples::TaggedTuple<gr::Tags::Shift<DataType, Dim>> variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<gr::Tags::Shift<DataType, Dim>> /*meta*/) const;

  template <typename DataType>
  tuples::TaggedTuple<::Tags::dt<gr::Tags::Shift<DataType, Dim>>> variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<::Tags::dt<gr::Tags::Shift<DataType, Dim>>> /*meta*/) const;

  template <typename DataType>
  tuples::TaggedTuple<::Tags::deriv<gr::Tags::Shift<DataType, Dim>,
                                    tmpl::size_t<Dim>, Frame::Inertial>>
  variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<::Tags::deriv<gr::Tags::Shift<DataType, Dim>,
                               tmpl::size_t<Dim>, Frame::Inertial>> /*meta*/)
      const;

  template <typename DataType>
  tuples::TaggedTuple<gr::Tags::SpatialMetric<DataType, Dim>> variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<gr::Tags::SpatialMetric<DataType, Dim>> /*meta*/) const;

  template <typename DataType>
  tuples::TaggedTuple<::Tags::dt<gr::Tags::SpatialMetric<DataType, Dim>>>
  variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<::Tags::dt<gr::Tags::SpatialMetric<DataType, Dim>>> /*meta*/)
      const;

  template <typename DataType>
  tuples::TaggedTuple<::Tags::deriv<gr::Tags::SpatialMetric<DataType, Dim>,
                                    tmpl::size_t<Dim>, Frame::Inertial>>
  variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<::Tags::deriv<gr::Tags::SpatialMetric<DataType, Dim>,
                               tmpl::size_t<Dim>, Frame::Inertial>> /*meta*/)
      const;

  template <typename DataType>
  tuples::TaggedTuple<gr::Tags::InverseSpatialMetric<DataType, Dim>> variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<gr::Tags::InverseSpatialMetric<DataType, Dim>> /*meta*/) const;

  template <typename DataType>
  tuples::TaggedTuple<gr::Tags::ExtrinsicCurvature<DataType, Dim>> variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<gr::Tags::ExtrinsicCurvature<DataType, Dim>> /*meta*/) const;

  template <typename DataType>
  tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataType>> variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>> /*meta*/) const;

  template <typename DataType>
  tuples::TaggedTuple<gr::Tags::DerivDetSpatialMetric<DataType, Dim>> variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<gr::Tags::DerivDetSpatialMetric<DataType, Dim>> /*meta*/)
      const;

  template <typename DataType>
  tuples::TaggedTuple<::Tags::dt<gr::Tags::SqrtDetSpatialMetric<DataType>>>
  variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<::Tags::dt<gr::Tags::SqrtDetSpatialMetric<DataType>>> /*meta*/)
      const;

  template <typename DataType>
  tuples::TaggedTuple<gr::Tags::TraceExtrinsicCurvature<DataType>> variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const;

  template <typename DataType>
  tuples::TaggedTuple<gr::Tags::SpatialChristoffelFirstKind<DataType, Dim>>
  variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<gr::Tags::SpatialChristoffelFirstKind<DataType, Dim>> /*meta*/)
      const;

  template <typename DataType>
  tuples::TaggedTuple<gr::Tags::SpatialChristoffelSecondKind<DataType, Dim>>
  variables(const tnsr::I<DataType, Dim>& x, double /*t*/,
            tmpl::list<
                gr::Tags::SpatialChristoffelSecondKind<DataType, Dim>> /*meta*/)
      const;

  template <typename DataType>
  tuples::TaggedTuple<
      gr::Tags::TraceSpatialChristoffelSecondKind<DataType, Dim>>
  variables(const tnsr::I<DataType, Dim>& x, double /*t*/,
            tmpl::list<gr::Tags::TraceSpatialChristoffelSecondKind<
                DataType, Dim>> /*meta*/) const;
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

template <size_t Dim>
inline constexpr bool operator==(const Minkowski<Dim>& /*lhs*/,
                                 const Minkowski<Dim>& /*rhs*/) {
  return true;
}

template <size_t Dim>
inline constexpr bool operator!=(const Minkowski<Dim>& /*lhs*/,
                                 const Minkowski<Dim>& /*rhs*/) {
  return false;
}
}  // namespace Solutions
}  // namespace gr
