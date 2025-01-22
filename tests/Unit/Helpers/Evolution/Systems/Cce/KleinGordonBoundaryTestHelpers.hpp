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

// Dump tensor+scalar data into a specified HDF5 file named `filename`.
// The tensor part comes from `AnalyticSolution` whereas the scalar part
// from `create_fake_time_varying_klein_gordon_data`.
template <typename AnalyticSolution>
void write_scalar_tensor_test_file(const AnalyticSolution& solution,
                                   const std::string& filename,
                                   const double target_time,
                                   const double extraction_radius,
                                   const double frequency,
                                   const double amplitude, const size_t l_max) {
  write_test_file(solution, filename, target_time, extraction_radius, frequency,
                  amplitude, l_max);

  const size_t goldberg_size = square(l_max + 1);
  Scalar<ComplexModalVector> kg_psi_modal{goldberg_size};
  Scalar<ComplexModalVector> kg_pi_modal{goldberg_size};
  Scalar<DataVector> kg_psi_nodal;
  Scalar<DataVector> kg_pi_nodal;

  // scoped to close the file
  {
    TestHelpers::WorldtubeModeRecorder recorder{l_max, filename};
    // write times to file for several steps before and after the target time
    for (size_t t = 0; t < 30; ++t) {
      const double time = 0.1 * static_cast<double>(t) + target_time - 1.5;
      // create scalar data
      TestHelpers::create_fake_time_varying_klein_gordon_data(
          make_not_null(&kg_psi_modal), make_not_null(&kg_pi_modal),
          make_not_null(&kg_psi_nodal), make_not_null(&kg_pi_nodal),
          extraction_radius, amplitude, frequency, time, l_max);

      // write scalar data
      recorder.append_worldtube_mode_data(
          detail::dataset_name_for_component("/KGPsi"), time, get(kg_psi_modal),
          false, true);
      recorder.append_worldtube_mode_data(
          detail::dataset_name_for_component("/dtKGPsi"), time,
          get(kg_pi_modal), false, true);
    }
  }
}
}  // namespace Cce::TestHelpers
