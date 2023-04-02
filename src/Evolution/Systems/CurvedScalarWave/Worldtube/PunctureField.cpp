// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/PunctureField.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DynamicBuffer.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {

void puncture_field(
    const gsl::not_null<Variables<tmpl::list<
        CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
        ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                      Frame::Inertial>>>*>
        result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords, const double time,
    const double orbital_radius, const double bh_mass, const size_t order) {
  if (order == 0) {
    puncture_field_0(result, coords, time, orbital_radius, bh_mass);
  } else if (order == 1) {
    puncture_field_1(result, coords, time, orbital_radius, bh_mass);
  } else if (order == 2) {
    puncture_field_2(result, coords, time, orbital_radius, bh_mass);
  } else {
    ERROR(
        "The puncture field is only implemented up to expansion order 2 but "
        "you requested order "
        << order);
  }
}
}  // namespace CurvedScalarWave::Worldtube
