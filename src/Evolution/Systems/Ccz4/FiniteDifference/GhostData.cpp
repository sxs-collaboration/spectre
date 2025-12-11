// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/GhostData.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
DataVector GhostVariables::apply(
    const Variables<Ccz4::fd::System::variables_tag_list>& evolved_vars,
    const size_t rdmp_size) {
  DataVector buffer{evolved_vars.number_of_grid_points() *
                        Variables<Ccz4::fd::System::variables_tag_list>::
                            number_of_independent_components +
                    rdmp_size};
  Variables<Ccz4::fd::System::variables_tag_list> vars_to_reconstruct(
      buffer.data(), buffer.size() - rdmp_size);

  get<Tags::ConformalMetric<DataVector, 3>>(vars_to_reconstruct) =
      get<Tags::ConformalMetric<DataVector, 3>>(evolved_vars);
  get<gr::Tags::Lapse<DataVector>>(vars_to_reconstruct) =
      get<gr::Tags::Lapse<DataVector>>(evolved_vars);
  get<gr::Tags::Shift<DataVector, 3>>(vars_to_reconstruct) =
      get<gr::Tags::Shift<DataVector, 3>>(evolved_vars);
  get<Tags::ConformalFactor<DataVector>>(vars_to_reconstruct) =
      get<Tags::ConformalFactor<DataVector>>(evolved_vars);
  get<Tags::ATilde<DataVector, 3>>(vars_to_reconstruct) =
      get<Tags::ATilde<DataVector, 3>>(evolved_vars);
  get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(vars_to_reconstruct) =
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(evolved_vars);
  get<Tags::Theta<DataVector>>(vars_to_reconstruct) =
      get<Tags::Theta<DataVector>>(evolved_vars);
  get<Tags::GammaHat<DataVector, 3>>(vars_to_reconstruct) =
      get<Tags::GammaHat<DataVector, 3>>(evolved_vars);
  get<Tags::AuxiliaryShiftB<DataVector, 3>>(vars_to_reconstruct) =
      get<Tags::AuxiliaryShiftB<DataVector, 3>>(evolved_vars);

  return buffer;
}
}  // namespace Ccz4::fd
