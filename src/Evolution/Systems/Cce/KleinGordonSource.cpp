// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/KleinGordonSource.hpp"

namespace Cce {

void ComputeKleinGordonSource<Tags::BondiBeta>::apply(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> kg_source_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& dy_kg_psi,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y) {
  get(*kg_source_beta) = 2. * M_PI * get(one_minus_y) * square(get(dy_kg_psi));
}

void ComputeKleinGordonSource<Tags::BondiQ>::apply(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> kg_source_q,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& dy_kg_psi,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_kg_psi) {
  get(*kg_source_q) = 16. * M_PI * get(eth_kg_psi) * get(dy_kg_psi);
}

void ComputeKleinGordonSource<Tags::BondiU>::apply(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> kg_source_u,
    const size_t l_max, const size_t number_of_radial_points) {
  const size_t volume_size =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
      number_of_radial_points;
  get(*kg_source_u) = SpinWeighted<ComplexDataVector, 1>{volume_size, 0.0};
}

void ComputeKleinGordonSource<Tags::BondiW>::apply(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> kg_source_w,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_kg_psi) {
  SpinWeighted<ComplexDataVector, 0> complex_part;
  SpinWeighted<ComplexDataVector, 0> real_part;

  complex_part.data() =
      2.0 * real(get(bondi_j).data() * square(conj(get(eth_kg_psi))).data());
  real_part.data() =
      -2.0 * get(bondi_k).data() * square(abs(get(eth_kg_psi).data()));
  get(*kg_source_w) =
      M_PI * get(exp_2_beta) / get(bondi_r) * (complex_part + real_part);
}

void ComputeKleinGordonSource<Tags::BondiH>::apply(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> kg_source_h,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_kg_psi) {
  get(*kg_source_h) =
      2 * M_PI * get(exp_2_beta) / get(bondi_r) * square(get(eth_kg_psi));
}
}  // namespace Cce
