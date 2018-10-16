// Distributed under the MIT License.
// See LICENSE.txt for details.

// IWYU pragma: no_include <array>

#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Characteristics.hpp"
#include "DataStructures/DataBox/Prefixes.hpp" // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp" // IWYU pragma: keep
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Tags::CharSpeed

namespace GeneralizedHarmonic {

template <size_t Dim, typename Frame>
typename Tags::CharacteristicSpeeds<Dim, Frame>::type
CharacteristicSpeedsCompute<Dim, Frame>::function(
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const tnsr::i<DataVector, Dim, Frame>& normal) noexcept {
  auto char_speeds =
      make_with_value<typename Tags::CharacteristicSpeeds<Dim, Frame>::type>(
          get(lapse), 0.);
  const auto shift_dot_normal = get(dot_product(shift, normal));
  get(get<::Tags::CharSpeed<Tags::UPsi<Dim, Frame>>>(char_speeds)) =
      -(1. + get(gamma_1)) * shift_dot_normal;
  get(get<::Tags::CharSpeed<Tags::UZero<Dim, Frame>>>(char_speeds)) =
      -shift_dot_normal;
  get(get<::Tags::CharSpeed<Tags::UPlus<Dim, Frame>>>(char_speeds)) =
      -shift_dot_normal + get(lapse);
  get(get<::Tags::CharSpeed<Tags::UMinus<Dim, Frame>>>(char_speeds)) =
      -shift_dot_normal - get(lapse);

  return char_speeds;
}

}  // namespace GeneralizedHarmonic

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data)                                      \
  template struct GeneralizedHarmonic::CharacteristicSpeedsCompute< \
      DIM(data), FRAME(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3),
                        (Frame::Inertial, Frame::Grid))

#undef INSTANTIATION
#undef DIM
