// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/Subcell/GhostData.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::subcell {
DataVector GhostVariables::apply(
    const Variables<evolved_vars>& vars,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_j,
    const size_t rdmp_size) {
  DataVector buffer{
      vars.number_of_grid_points() *
          (tilde_j.size() +
           Variables<evolved_vars>::number_of_independent_components) +
      rdmp_size};

  Variables<tmpl::append<tmpl::list<ForceFree::Tags::TildeJ>, evolved_vars>>
      vars_to_reconstruct(buffer.data(), buffer.size() - rdmp_size);

  for (size_t i = 0; i < 3; ++i) {
    get<ForceFree::Tags::TildeJ>(vars_to_reconstruct).get(i) = tilde_j.get(i);
  }

  tmpl::for_each<evolved_vars>([&vars, &vars_to_reconstruct](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    get<tag>(vars_to_reconstruct) = get<tag>(vars);
  });

  return buffer;
}
}  // namespace ForceFree::subcell
