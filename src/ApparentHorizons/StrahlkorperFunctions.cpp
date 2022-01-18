// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/StrahlkorperFunctions.hpp"

#include "ApparentHorizons/Strahlkorper.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace StrahlkorperFunctions {
template <typename Fr>
Scalar<DataVector> radius(const Strahlkorper<Fr>& strahlkorper) {
  Scalar<DataVector> result{
      DataVector{strahlkorper.ylm_spherepack().physical_size()}};
  radius(make_not_null(&result), strahlkorper);
  return result;
}

template <typename Fr>
void radius(const gsl::not_null<Scalar<DataVector>*> result,
            const Strahlkorper<Fr>& strahlkorper) {
  get(*result) =
      strahlkorper.ylm_spherepack().spec_to_phys(strahlkorper.coefficients());
}
}  // namespace StrahlkorperFunctions

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                 \
  template Scalar<DataVector> StrahlkorperFunctions::radius( \
      const Strahlkorper<FRAME(data)>& strahlkorper);        \
  template void StrahlkorperFunctions::radius(               \
      const gsl::not_null<Scalar<DataVector>*> result,       \
      const Strahlkorper<FRAME(data)>& strahlkorper);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Distorted, Frame::Grid, Frame::Inertial))

#undef INSTANTIATE
#undef FRAME
