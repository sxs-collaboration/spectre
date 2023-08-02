// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Surfaces/AreaElement.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace StrahlkorperGr {
template <typename Frame>
void area_element(const gsl::not_null<Scalar<DataVector>*> result,
                  const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
                  const StrahlkorperTags::aliases::Jacobian<Frame>& jacobian,
                  const tnsr::i<DataVector, 3, Frame>& normal_one_form,
                  const Scalar<DataVector>& radius,
                  const tnsr::i<DataVector, 3, Frame>& r_hat) {
  auto cap_theta = make_with_value<tnsr::I<DataVector, 3, Frame>>(r_hat, 0.0);
  auto cap_phi = make_with_value<tnsr::I<DataVector, 3, Frame>>(r_hat, 0.0);

  for (size_t i = 0; i < 3; ++i) {
    cap_theta.get(i) = jacobian.get(i, 0);
    cap_phi.get(i) = jacobian.get(i, 1);
    for (size_t j = 0; j < 3; ++j) {
      cap_theta.get(i) += r_hat.get(i) *
                          (r_hat.get(j) - normal_one_form.get(j)) *
                          jacobian.get(j, 0);
      cap_phi.get(i) += r_hat.get(i) * (r_hat.get(j) - normal_one_form.get(j)) *
                        jacobian.get(j, 1);
    }
  }

  get(*result) = square(get(radius));
  get(*result) *=
      sqrt(get(dot_product(cap_theta, cap_theta, spatial_metric)) *
               get(dot_product(cap_phi, cap_phi, spatial_metric)) -
           square(get(dot_product(cap_theta, cap_phi, spatial_metric))));
}

template <typename Frame>
Scalar<DataVector> area_element(
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const StrahlkorperTags::aliases::Jacobian<Frame>& jacobian,
    const tnsr::i<DataVector, 3, Frame>& normal_one_form,
    const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, 3, Frame>& r_hat) {
  Scalar<DataVector> result{};
  area_element(make_not_null(&result), spatial_metric, jacobian,
               normal_one_form, radius, r_hat);
  return result;
}

template <typename Frame>
void euclidean_area_element(
    const gsl::not_null<Scalar<DataVector>*> result,
    const StrahlkorperTags::aliases::Jacobian<Frame>& jacobian,
    const tnsr::i<DataVector, 3, Frame>& normal_one_form,
    const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, 3, Frame>& r_hat) {
  auto cap_theta = make_with_value<tnsr::I<DataVector, 3, Frame>>(r_hat, 0.0);
  auto cap_phi = make_with_value<tnsr::I<DataVector, 3, Frame>>(r_hat, 0.0);

  for (size_t i = 0; i < 3; ++i) {
    cap_theta.get(i) = jacobian.get(i, 0);
    cap_phi.get(i) = jacobian.get(i, 1);
    for (size_t j = 0; j < 3; ++j) {
      cap_theta.get(i) += r_hat.get(i) *
                          (r_hat.get(j) - normal_one_form.get(j)) *
                          jacobian.get(j, 0);
      cap_phi.get(i) += r_hat.get(i) * (r_hat.get(j) - normal_one_form.get(j)) *
                        jacobian.get(j, 1);
    }
  }

  get(*result) = square(get(radius));
  get(*result) *= sqrt(get(dot_product(cap_theta, cap_theta)) *
                           get(dot_product(cap_phi, cap_phi)) -
                       square(get(dot_product(cap_theta, cap_phi))));
}

template <typename Frame>
Scalar<DataVector> euclidean_area_element(
    const StrahlkorperTags::aliases::Jacobian<Frame>& jacobian,
    const tnsr::i<DataVector, 3, Frame>& normal_one_form,
    const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, 3, Frame>& r_hat) {
  Scalar<DataVector> result{};
  euclidean_area_element(make_not_null(&result), jacobian, normal_one_form,
                         radius, r_hat);
  return result;
}
}  // namespace StrahlkorperGr

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                             \
  template void StrahlkorperGr::area_element<FRAME(data)>(               \
      const gsl::not_null<Scalar<DataVector>*> result,                   \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_metric,        \
      const StrahlkorperTags::aliases::Jacobian<FRAME(data)>& jacobian,  \
      const tnsr::i<DataVector, 3, FRAME(data)>& normal_one_form,        \
      const Scalar<DataVector>& radius,                                  \
      const tnsr::i<DataVector, 3, FRAME(data)>& r_hat);                 \
  template Scalar<DataVector> StrahlkorperGr::area_element<FRAME(data)>( \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_metric,        \
      const StrahlkorperTags::aliases::Jacobian<FRAME(data)>& jacobian,  \
      const tnsr::i<DataVector, 3, FRAME(data)>& normal_one_form,        \
      const Scalar<DataVector>& radius,                                  \
      const tnsr::i<DataVector, 3, FRAME(data)>& r_hat);                 \
  template void StrahlkorperGr::euclidean_area_element<FRAME(data)>(     \
      const gsl::not_null<Scalar<DataVector>*> result,                   \
      const StrahlkorperTags::aliases::Jacobian<FRAME(data)>& jacobian,  \
      const tnsr::i<DataVector, 3, FRAME(data)>& normal_one_form,        \
      const Scalar<DataVector>& radius,                                  \
      const tnsr::i<DataVector, 3, FRAME(data)>& r_hat);                 \
  template Scalar<DataVector>                                            \
  StrahlkorperGr::euclidean_area_element<FRAME(data)>(                   \
      const StrahlkorperTags::aliases::Jacobian<FRAME(data)>& jacobian,  \
      const tnsr::i<DataVector, 3, FRAME(data)>& normal_one_form,        \
      const Scalar<DataVector>& radius,                                  \
      const tnsr::i<DataVector, 3, FRAME(data)>& r_hat);
GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))
#undef INSTANTIATE
#undef FRAME
