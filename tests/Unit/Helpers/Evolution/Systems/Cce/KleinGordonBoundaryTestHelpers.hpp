// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Helpers/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"

namespace Cce::TestHelpers {

// The nodal data for the scalar field psi reads
//
// psi = sin(r - t)
//
// where r is a time-dependent radius
//
// r = (1 + A * sin ft) * R
//
// Its time derivative is given by
//
// dr/dt = A * f * cos ft * R
//       =  r / (1 + A * sin ft) * A * f * cos ft
void create_fake_time_varying_klein_gordon_data(
    gsl::not_null<Scalar<ComplexModalVector>*> kg_psi_modal,
    gsl::not_null<Scalar<ComplexModalVector>*> kg_pi_modal,
    gsl::not_null<Scalar<DataVector>*> kg_psi_nodal,
    gsl::not_null<Scalar<DataVector>*> kg_pi_nodal, double extraction_radius,
    double amplitude, double frequency, double time, size_t l_max);

}  // namespace Cce::TestHelpers
