// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/SpatialCurrentDensity.hpp"

#include <cstddef>

#include "DataStructures/Blaze/StepFunction.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree {

void SpatialCurrentDensity::apply(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        spatial_current_density,
    const Scalar<DataVector>& tilde_q,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const double parallel_conductivity,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric) {
  Variables<tmpl::list<::Tags::Tempi<0, 3>, ::Tags::Tempi<1, 3>,
                       ::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>>>
      buffer{get(tilde_q).size()};

  // compute \tilde{E}_j, \tilde{B}_j
  auto& tilde_e_one_form = get<::Tags::Tempi<0, 3>>(buffer);
  auto& tilde_b_one_form = get<::Tags::Tempi<1, 3>>(buffer);
  raise_or_lower_index(make_not_null(&tilde_e_one_form), tilde_e,
                       spatial_metric);
  raise_or_lower_index(make_not_null(&tilde_b_one_form), tilde_b,
                       spatial_metric);

  // compute  \tilde{E}^2 = \tilde{E}_j \tilde{E}^j
  //     and  \tilde{B}^2 = \tilde{B}^j \tilde{B}_j
  //     and  \tilde{E}^j\tilde{B}_j
  auto& tilde_e_squared = get<::Tags::TempScalar<0>>(buffer);
  auto& tilde_b_squared = get<::Tags::TempScalar<1>>(buffer);
  auto& tilde_e_dot_tilde_b = get<::Tags::TempScalar<2>>(buffer);
  dot_product(make_not_null(&tilde_e_squared), tilde_e_one_form, tilde_e);
  dot_product(make_not_null(&tilde_b_squared), tilde_b_one_form, tilde_b);
  dot_product(make_not_null(&tilde_e_dot_tilde_b), tilde_e_one_form, tilde_b);

  // relaxation term for the force-free constraints; note that in the future
  // this term should be treated implicitly using the IMEX time integration
  for (size_t i = 0; i < 3; ++i) {
    (*spatial_current_density).get(i) =
        parallel_conductivity *
        (get(tilde_e_dot_tilde_b) * tilde_b.get(i) +
         max(get(tilde_e_squared) - get(tilde_b_squared), 0.0) *
             tilde_e.get(i));
  }

  // drift current term
  for (LeviCivitaIterator<3> it; it; ++it) {
    const auto& i = it[0];
    const auto& j = it[1];
    const auto& k = it[2];

    (*spatial_current_density).get(i) +=
        it.sign() * get(tilde_q) * tilde_e_one_form.get(j) *
        tilde_b_one_form.get(k) /
        get(sqrt_det_spatial_metric);  // the extra 1/sqrt{gamma} factor comes
                                       // from the Levi-Civita tensor
  }

  // scale out a common factor
  for (size_t i = 0; i < 3; ++i) {
    (*spatial_current_density).get(i) /=
        get(sqrt_det_spatial_metric) * get(tilde_b_squared);
  }
}

}  // namespace ForceFree
