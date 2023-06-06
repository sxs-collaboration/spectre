// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GeneralRelativity/BrillLindquist.hpp"

#include <array>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/ParseError.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace gr::AnalyticData {
BrillLindquist::BrillLindquist(const double mass_a, const double mass_b,
                               const std::array<double, 3>& center_a,
                               const std::array<double, 3>& center_b,
                               const Options::Context& context)
    : mass_a_(mass_a),
      mass_b_(mass_b),
      center_a_(center_a),
      center_b_(center_b) {
  if (mass_a_ <= 0.0) {
    PARSE_ERROR(context,
                "Mass A must be positive. Given mass: " << mass_a_);
  }
  if (mass_b_ <= 0.0) {
    PARSE_ERROR(context,
                "Mass B must be positive. Given mass: " << mass_b_);
  }
}

BrillLindquist::BrillLindquist(CkMigrateMessage* /*unused*/) {}

void BrillLindquist::pup(PUP::er& p) {
  p | mass_a_;
  p | mass_b_;
  p | center_a_;
  p | center_b_;
}

template <typename DataType, typename Frame>
BrillLindquist::IntermediateComputer<DataType, Frame>::IntermediateComputer(
    const BrillLindquist& analytic_data, const tnsr::I<DataType, 3, Frame>& x)
    : analytic_data_(analytic_data), x_(x) {}

template <typename DataType, typename Frame>
void BrillLindquist::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_minus_center_a,
    const gsl::not_null<CachedBuffer*> /*cache*/,
    internal_tags::x_minus_center_a<DataType, Frame> /*meta*/) const {
  for (size_t i = 0; i < 3; ++i) {
    x_minus_center_a->get(i) =
        x_.get(i) - gsl::at(analytic_data_.center_a(), i);
  }
}

template <typename DataType, typename Frame>
void BrillLindquist::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r_a,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r_a<DataType> /*meta*/) const {
  const auto& x_minus_center_a =
      cache->get_var(*this, internal_tags::x_minus_center_a<DataType, Frame>{});
  magnitude(r_a, x_minus_center_a);
}

template <typename DataType, typename Frame>
void BrillLindquist::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_minus_center_b,
    const gsl::not_null<CachedBuffer*> /*cache*/,
    internal_tags::x_minus_center_b<DataType, Frame> /*meta*/) const {
  for (size_t i = 0; i < 3; ++i) {
    x_minus_center_b->get(i) =
        x_.get(i) - gsl::at(analytic_data_.center_b(), i);
  }
}

template <typename DataType, typename Frame>
void BrillLindquist::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r_b,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r_b<DataType> /*meta*/) const {
  const auto& x_minus_center_b =
      cache->get_var(*this, internal_tags::x_minus_center_b<DataType, Frame>{});
  magnitude(r_b, x_minus_center_b);
}

template <typename DataType, typename Frame>
void BrillLindquist::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_factor,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::conformal_factor<DataType> /*meta*/) const {
  const auto& r_a = cache->get_var(*this, internal_tags::r_a<DataType>{});
  const auto& r_b = cache->get_var(*this, internal_tags::r_b<DataType>{});
  get(*conformal_factor) = 1.0 + 0.5 * analytic_data_.mass_a() / get(r_a) +
                           0.5 * analytic_data_.mass_b() / get(r_b);
}

template <typename DataType, typename Frame>
void BrillLindquist::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3, Frame>*> deriv_conformal_factor,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_conformal_factor<DataType, Frame> /*meta*/) const {
  const auto& r_a = cache->get_var(*this, internal_tags::r_a<DataType>{});
  const auto& r_b = cache->get_var(*this, internal_tags::r_b<DataType>{});
  const auto& x_minus_center_a =
      cache->get_var(*this, internal_tags::x_minus_center_a<DataType, Frame>{});
  const auto& x_minus_center_b =
      cache->get_var(*this, internal_tags::x_minus_center_b<DataType, Frame>{});
  for (size_t i = 0; i < 3; ++i) {
    deriv_conformal_factor->get(i) =
        -0.5 * analytic_data_.mass_a() * x_minus_center_a.get(i) /
            cube(get(r_a)) -
        0.5 * analytic_data_.mass_b() * x_minus_center_b.get(i) /
            cube(get(r_b));
  }
}

template <typename DataType, typename Frame>
void BrillLindquist::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> spatial_metric,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::SpatialMetric<DataType, 3, Frame> /*meta*/) const {
  const auto& conformal_factor =
      cache->get_var(*this, internal_tags::conformal_factor<DataType>{});
  std::fill(spatial_metric->begin(), spatial_metric->end(), 0.);
  get<0, 0>(*spatial_metric) = pow<4>(get(conformal_factor));
  get<1, 1>(*spatial_metric) = get<0, 0>(*spatial_metric);
  get<2, 2>(*spatial_metric) = get<0, 0>(*spatial_metric);
}

template <typename DataType, typename Frame>
void BrillLindquist::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, 3, Frame>*> deriv_spatial_metric,
    const gsl::not_null<CachedBuffer*> cache,
    DerivSpatialMetric<DataType, Frame> /*meta*/) const {
  const auto& conformal_factor =
      cache->get_var(*this, internal_tags::conformal_factor<DataType>{});
  const auto& deriv_conformal_factor = cache->get_var(
      *this, internal_tags::deriv_conformal_factor<DataType, Frame>{});
  std::fill(deriv_spatial_metric->begin(), deriv_spatial_metric->end(), 0.);
  for (size_t k = 0; k < 3; ++k) {
    deriv_spatial_metric->get(k, 0, 0) =
        4.0 * pow<3>(get(conformal_factor)) * deriv_conformal_factor.get(k);
    deriv_spatial_metric->get(k, 1, 1) = deriv_spatial_metric->get(k, 0, 0);
    deriv_spatial_metric->get(k, 2, 2) = deriv_spatial_metric->get(k, 0, 0);
  }
}

template <typename DataType, typename Frame>
Scalar<DataType> BrillLindquist::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::Lapse<DataType> /*meta*/) {
  const auto& r_a = get(get_var(computer, internal_tags::r_a<DataType>{}));
  return make_with_value<Scalar<DataType>>(r_a, 1.);
}

template <typename DataType, typename Frame>
Scalar<DataType> BrillLindquist::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    ::Tags::dt<gr::Tags::Lapse<DataType>> /*meta*/) {
  const auto& r_a = get(get_var(computer, internal_tags::r_a<DataType>{}));
  return make_with_value<Scalar<DataType>>(r_a, 0.);
}

template <typename DataType, typename Frame>
tnsr::i<DataType, 3, Frame>
BrillLindquist::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    DerivLapse<DataType, Frame> /*meta*/) {
  const auto& r_a = get(get_var(computer, internal_tags::r_a<DataType>{}));
  return make_with_value<tnsr::i<DataType, 3, Frame>>(r_a, 0.);
}

template <typename DataType, typename Frame>
tnsr::I<DataType, 3, Frame>
BrillLindquist::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::Shift<DataType, 3, Frame> /*meta*/) {
  const auto& r_a = get(get_var(computer, internal_tags::r_a<DataType>{}));
  return make_with_value<tnsr::I<DataType, 3, Frame>>(r_a, 0.);
}

template <typename DataType, typename Frame>
tnsr::I<DataType, 3, Frame>
BrillLindquist::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    ::Tags::dt<gr::Tags::Shift<DataType, 3, Frame>> /*meta*/) {
  const auto& r_a = get(get_var(computer, internal_tags::r_a<DataType>{}));
  return make_with_value<tnsr::I<DataType, 3, Frame>>(r_a, 0.);
}

template <typename DataType, typename Frame>
tnsr::iJ<DataType, 3, Frame>
BrillLindquist::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    DerivShift<DataType, Frame> /*meta*/) {
  const auto& r_a = get(get_var(computer, internal_tags::r_a<DataType>{}));
  return make_with_value<tnsr::iJ<DataType, 3, Frame>>(r_a, 0.);
}

template <typename DataType, typename Frame>
tnsr::ii<DataType, 3, Frame>
BrillLindquist::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    ::Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame>> /*meta*/) {
  const auto& r_a = get(get_var(computer, internal_tags::r_a<DataType>{}));
  return make_with_value<tnsr::ii<DataType, 3, Frame>>(r_a, 0.);
}

template <typename DataType, typename Frame>
Scalar<DataType> BrillLindquist::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::SqrtDetSpatialMetric<DataType> /*meta*/) {
  const auto& conformal_factor =
      get(get_var(computer, internal_tags::conformal_factor<DataType>{}));
  return Scalar<DataType>(pow<6>(conformal_factor));
}

template <typename DataType, typename Frame>
tnsr::II<DataType, 3, Frame>
BrillLindquist::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::InverseSpatialMetric<DataType, 3, Frame> /*meta*/) {
  const auto& spatial_metric =
      get_var(computer, gr::Tags::SpatialMetric<DataType, 3, Frame>{});
  tnsr::II<DataType, 3, Frame> inverse_spatial_metric{};
  get<0, 0>(inverse_spatial_metric) = 1.0 / get<0, 0>(spatial_metric);
  get<1, 1>(inverse_spatial_metric) = 1.0 / get<1, 1>(spatial_metric);
  get<2, 2>(inverse_spatial_metric) = 1.0 / get<2, 2>(spatial_metric);
  return inverse_spatial_metric;
}

template <typename DataType, typename Frame>
tnsr::ii<DataType, 3, Frame>
BrillLindquist::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::ExtrinsicCurvature<DataType, 3, Frame> /*meta*/) {
  const auto& r_a = get(get_var(computer, internal_tags::r_a<DataType>{}));
  return make_with_value<tnsr::ii<DataType, 3, Frame>>(r_a, 0.);
}

bool operator==(const BrillLindquist& lhs, const BrillLindquist& rhs) {
  return lhs.mass_a() == rhs.mass_a() and lhs.mass_b() == rhs.mass_b() and
         lhs.center_a() == rhs.center_a() and lhs.center_b() == rhs.center_b();
}

bool operator!=(const BrillLindquist& lhs, const BrillLindquist& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                 \
  template class BrillLindquist::IntermediateVars<DTYPE(data), FRAME(data)>; \
  template class BrillLindquist::IntermediateComputer<DTYPE(data), FRAME(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector, double),
                        (::Frame::Inertial, ::Frame::Grid))
#undef INSTANTIATE
#undef DTYPE
#undef FRAME
}  // namespace gr::AnalyticData
