// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Actions/ScriObserveInterpolated.hpp"

namespace Cce::Actions::detail {
void correct_weyl_scalars_for_inertial_time(
    const gsl::not_null<Variables<weyl_correction_list>*>
        weyl_correction_variables) noexcept {
  const auto& psi_4 =
      get<Tags::Du<Tags::TimeIntegral<Tags::ScriPlus<Tags::Psi4>>>>(
          *weyl_correction_variables);
  auto& psi_3 = get<Tags::ScriPlus<Tags::Psi3>>(*weyl_correction_variables);
  auto& psi_2 = get<Tags::ScriPlus<Tags::Psi2>>(*weyl_correction_variables);
  auto& psi_1 = get<Tags::ScriPlus<Tags::Psi1>>(*weyl_correction_variables);
  auto& psi_0 = get<Tags::ScriPlus<Tags::Psi0>>(*weyl_correction_variables);
  // note the variable `eth_u` corresponds to $\eth u^\prime$ in the
  // documentation
  const auto& eth_u =
      get<Tags::EthInertialRetardedTime>(*weyl_correction_variables);
  get(psi_0) += 2.0 * get(eth_u) * get(psi_1) +
               0.75 * square(get(eth_u)) * get(psi_2) +
               0.5 * pow<3>(get(eth_u)) * get(psi_3) +
               0.0625 * pow<4>(get(eth_u)) * get(psi_4);
  get(psi_1) += 1.5 * get(eth_u) * get(psi_2) +
                0.75 * square(get(eth_u)) * get(psi_3) +
                0.125 * pow<3>(get(eth_u)) * get(psi_4);
  get(psi_2) +=
      get(eth_u) * get(psi_3) + 0.25 * square(get(eth_u)) * get(psi_4);
  get(psi_3) += 0.5 * get(eth_u) * get(psi_4);
}
}  // namespace Cce::Actions::detail
