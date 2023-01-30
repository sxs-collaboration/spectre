// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Psi4.hpp"

#include <cstddef>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylPropagating.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace gr {
template <typename Frame>
void psi_4(const gsl::not_null<Scalar<ComplexDataVector>*> psi_4_result,
           const tnsr::ii<DataVector, 3, Frame>& spatial_ricci,
           const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
           const tnsr::ijj<DataVector, 3, Frame>& cov_deriv_extrinsic_curvature,
           const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
           const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric,
           const tnsr::I<DataVector, 3, Frame>& inertial_coords) {
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempI<0, 3, Frame>,
                       ::Tags::TempI<1, 3, Frame>, ::Tags::TempI<2, 3, Frame>,
                       ::Tags::TempI<3, 3, Frame>, ::Tags::Tempi<0, 3, Frame>,
                       ::Tags::Tempij<0, 3, Frame>, ::Tags::Tempii<0, 3, Frame>,
                       ::Tags::Tempii<1, 3, Frame>, ::Tags::TempIj<0, 3, Frame>,
                       ::Tags::TempII<0, 3, Frame>>>
      temp_buffer{get<0>(inertial_coords).size()};
  auto& magnitude_cartesian = get<::Tags::TempScalar<0>>(temp_buffer);
  magnitude(make_not_null(&magnitude_cartesian), inertial_coords,
            spatial_metric);
  auto& r_hat = get<::Tags::TempI<0, 3, Frame>>(temp_buffer);
  for (size_t j = 0; j < 3; j++) {
    for (size_t i = 0; i < get(magnitude_cartesian).size(); i++) {
      if (magnitude_cartesian.get()[i] != 0.0) {
        r_hat.get(j)[i] =
            inertial_coords.get(j)[i] / magnitude_cartesian.get()[i];
      } else {
        r_hat.get(j)[i] = 0.0;
      }
    }
  }
  auto& lower_r_hat = get<::Tags::Tempi<0, 3, Frame>>(temp_buffer);
  tenex::evaluate<ti::i>(make_not_null(&lower_r_hat),
                         r_hat(ti::J) * spatial_metric(ti::i, ti::j));
  auto& projection_tensor = get<::Tags::Tempii<0, 3, Frame>>(temp_buffer);
  transverse_projection_operator(make_not_null(&projection_tensor),
                                 spatial_metric, lower_r_hat);
  auto& inverse_projection_tensor =
      get<::Tags::TempII<0, 3, Frame>>(temp_buffer);
  transverse_projection_operator(make_not_null(&inverse_projection_tensor),
                                 inverse_spatial_metric, r_hat);
  auto& projection_up_lo = get<::Tags::TempIj<0, 3, Frame>>(temp_buffer);
  tenex::evaluate<ti::K, ti::i>(
      make_not_null(&projection_up_lo),
      projection_tensor(ti::i, ti::j) * inverse_spatial_metric(ti::K, ti::J));

  auto& u8_plus = get<::Tags::Tempii<1, 3, Frame>>(temp_buffer);
  gr::weyl_propagating(
      make_not_null(&u8_plus), spatial_ricci, extrinsic_curvature,
      inverse_spatial_metric, cov_deriv_extrinsic_curvature, r_hat,
      inverse_projection_tensor, projection_tensor, projection_up_lo, 1.0);

  // Gram-Schmidt x_hat, a unit vector that's orthogonal to r_hat
  auto& x_coord = get<::Tags::TempI<1, 3, Frame>>(temp_buffer);
  x_coord.get(0) = 1.0;
  x_coord.get(1) = x_coord.get(2) = 0.0;
  auto& x_component = get<::Tags::TempScalar<0>>(temp_buffer);
  dot_product(make_not_null(&x_component), x_coord, r_hat, spatial_metric);
  auto& x_hat = get<::Tags::TempI<2, 3, Frame>>(temp_buffer);
  tenex::evaluate<ti::I>(make_not_null(&x_hat),
                         x_coord(ti::I) - (x_component() * r_hat(ti::I)));
  auto& magnitude_x = get<::Tags::TempScalar<0>>(temp_buffer);
  magnitude(make_not_null(&magnitude_x), x_hat, spatial_metric);
  for (size_t j = 0; j < 3; j++) {
    for (size_t i = 0; i < get(magnitude_x).size(); i++) {
      if (magnitude_x.get()[i] != 0.0) {
        x_hat.get(j)[i] /= magnitude_x.get()[i];
      } else {
        x_hat.get(j)[i] = 0.0;
      }
    }
  }

  Variables<tmpl::list<::Tags::TempI<0, 3, Frame, ComplexDataVector>,
                       ::Tags::TempI<1, 3, Frame, ComplexDataVector>>>
      y_hat_buffer{get<0>(inertial_coords).size()};

  // Grad-Schmidt y_hat, a unit vector orthogonal to r_hat and x_hat
  auto& y_coord = get<::Tags::TempI<1, 3, Frame>>(temp_buffer);
  y_coord.get(1) = 1.0;
  y_coord.get(0) = y_coord.get(2) = 0.0;
  auto& y_component = get<::Tags::TempScalar<0>>(temp_buffer);
  dot_product(make_not_null(&y_component), y_coord, r_hat, spatial_metric);
  auto& y_hat_not_complex = get<::Tags::TempI<3, 3, Frame>>(temp_buffer);
  tenex::evaluate<ti::I>(make_not_null(&y_hat_not_complex),
                         y_coord(ti::I) - (y_component() * r_hat(ti::I)));
  dot_product(make_not_null(&y_component), y_coord, x_hat, spatial_metric);
  tenex::evaluate<ti::I>(
      make_not_null(&y_hat_not_complex),
      y_hat_not_complex(ti::I) - (y_component() * x_hat(ti::I)));
  auto& magnitude_y = get<::Tags::TempScalar<0>>(temp_buffer);
  magnitude(make_not_null(&magnitude_y), y_hat_not_complex, spatial_metric);
  for (size_t j = 0; j < 3; j++) {
    for (size_t i = 0; i < get(magnitude_y).size(); i++) {
      if (magnitude_y.get()[i] != 0.0) {
        y_hat_not_complex.get(j)[i] /= magnitude_y.get()[i];
      } else {
        y_hat_not_complex.get(j)[i] = 0.0;
      }
    }
  }
  const std::complex<double> imag = std::complex<double>(0.0, 1.0);
  auto& y_hat =
      get<::Tags::TempI<0, 3, Frame, ComplexDataVector>>(y_hat_buffer);
  tenex::evaluate<ti::I>(make_not_null(&y_hat),
                         imag * y_hat_not_complex(ti::I));

  auto& m_bar =
      get<::Tags::TempI<1, 3, Frame, ComplexDataVector>>(y_hat_buffer);
  tenex::evaluate<ti::I>(make_not_null(&m_bar), x_hat(ti::I) - y_hat(ti::I));

  tenex::evaluate(psi_4_result,
                  -0.5 * u8_plus(ti::i, ti::j) * m_bar(ti::I) * m_bar(ti::J));
}

template <typename Frame>
Scalar<ComplexDataVector> psi_4(
    const tnsr::ii<DataVector, 3, Frame>& spatial_ricci,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
    const tnsr::ijj<DataVector, 3, Frame>& cov_deriv_extrinsic_curvature,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric,
    const tnsr::I<DataVector, 3, Frame>& inertial_coords) {
  auto psi_4_result = make_with_value<Scalar<ComplexDataVector>>(
      get<0, 0>(inverse_spatial_metric),
      std::numeric_limits<double>::signaling_NaN());
  psi_4(make_not_null(&psi_4_result), spatial_ricci, extrinsic_curvature,
        cov_deriv_extrinsic_curvature, spatial_metric, inverse_spatial_metric,
        inertial_coords);
  return psi_4_result;
}
}  // namespace gr

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                              \
  template Scalar<ComplexDataVector> gr::psi_4(                           \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_ricci,          \
      const tnsr::ii<DataVector, 3, FRAME(data)>& extrinsic_curvature,    \
      const tnsr::ijj<DataVector, 3, FRAME(data)>&                        \
          cov_deriv_extrinsic_curvature,                                  \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_metric,         \
      const tnsr::II<DataVector, 3, FRAME(data)>& inverse_spatial_metric, \
      const tnsr::I<DataVector, 3, FRAME(data)>& inertial_coords);        \
  template void gr::psi_4(                                                \
      const gsl::not_null<Scalar<ComplexDataVector>*> psi_4_result,       \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_ricci,          \
      const tnsr::ii<DataVector, 3, FRAME(data)>& extrinsic_curvature,    \
      const tnsr::ijj<DataVector, 3, FRAME(data)>&                        \
          cov_deriv_extrinsic_curvature,                                  \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_metric,         \
      const tnsr::II<DataVector, 3, FRAME(data)>& inverse_spatial_metric, \
      const tnsr::I<DataVector, 3, FRAME(data)>& inertial_coords);

GENERATE_INSTANTIATIONS(INSTANTIATE, (Frame::Grid, Frame::Inertial))

#undef FRAME
#undef INSTANTIATE
