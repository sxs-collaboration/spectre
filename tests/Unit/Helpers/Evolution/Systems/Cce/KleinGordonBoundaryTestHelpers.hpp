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
  const size_t goldberg_size = square(l_max + 1);
  tnsr::ii<ComplexModalVector, 3> spatial_metric_coefficients{goldberg_size};
  tnsr::ii<ComplexModalVector, 3> dt_spatial_metric_coefficients{goldberg_size};
  tnsr::ii<ComplexModalVector, 3> dr_spatial_metric_coefficients{goldberg_size};
  tnsr::I<ComplexModalVector, 3> shift_coefficients{goldberg_size};
  tnsr::I<ComplexModalVector, 3> dt_shift_coefficients{goldberg_size};
  tnsr::I<ComplexModalVector, 3> dr_shift_coefficients{goldberg_size};
  Scalar<ComplexModalVector> lapse_coefficients{goldberg_size};
  Scalar<ComplexModalVector> dt_lapse_coefficients{goldberg_size};
  Scalar<ComplexModalVector> dr_lapse_coefficients{goldberg_size};

  Scalar<ComplexModalVector> kg_psi_modal{goldberg_size};
  Scalar<ComplexModalVector> kg_pi_modal{goldberg_size};
  Scalar<DataVector> kg_psi_nodal;
  Scalar<DataVector> kg_pi_nodal;

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  // scoped to close the file
  {
    TestHelpers::WorldtubeModeRecorder recorder{filename, l_max};
    // write times to file for several steps before and after the target time
    for (size_t t = 0; t < 30; ++t) {
      const double time = 0.1 * static_cast<double>(t) + target_time - 1.5;
      // create tensor data
      TestHelpers::create_fake_time_varying_modal_data(
          make_not_null(&spatial_metric_coefficients),
          make_not_null(&dt_spatial_metric_coefficients),
          make_not_null(&dr_spatial_metric_coefficients),
          make_not_null(&shift_coefficients),
          make_not_null(&dt_shift_coefficients),
          make_not_null(&dr_shift_coefficients),
          make_not_null(&lapse_coefficients),
          make_not_null(&dt_lapse_coefficients),
          make_not_null(&dr_lapse_coefficients), solution, extraction_radius,
          amplitude, frequency, time, l_max);

      // create scalar data
      TestHelpers::create_fake_time_varying_klein_gordon_data(
          make_not_null(&kg_psi_modal), make_not_null(&kg_pi_modal),
          make_not_null(&kg_psi_nodal), make_not_null(&kg_pi_nodal),
          extraction_radius, amplitude, frequency, time, l_max);

      // write tensor data
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i; j < 3; ++j) {
          recorder.append_worldtube_mode_data(
              detail::dataset_name_for_component("/g", i, j), time,
              spatial_metric_coefficients.get(i, j), l_max);
          recorder.append_worldtube_mode_data(
              detail::dataset_name_for_component("/Drg", i, j), time,
              dr_spatial_metric_coefficients.get(i, j), l_max);
          recorder.append_worldtube_mode_data(
              detail::dataset_name_for_component("/Dtg", i, j), time,
              dt_spatial_metric_coefficients.get(i, j), l_max);
        }
        recorder.append_worldtube_mode_data(
            detail::dataset_name_for_component("/Shift", i), time,
            shift_coefficients.get(i), l_max);
        recorder.append_worldtube_mode_data(
            detail::dataset_name_for_component("/DrShift", i), time,
            dr_shift_coefficients.get(i), l_max);
        recorder.append_worldtube_mode_data(
            detail::dataset_name_for_component("/DtShift", i), time,
            dt_shift_coefficients.get(i), l_max);
      }
      recorder.append_worldtube_mode_data(
          detail::dataset_name_for_component("/Lapse"), time,
          get(lapse_coefficients), l_max);
      recorder.append_worldtube_mode_data(
          detail::dataset_name_for_component("/DrLapse"), time,
          get(dr_lapse_coefficients), l_max);
      recorder.append_worldtube_mode_data(
          detail::dataset_name_for_component("/DtLapse"), time,
          get(dt_lapse_coefficients), l_max);

      // write scalar data
      recorder.append_worldtube_mode_data(
          detail::dataset_name_for_component("/KGPsi"), time, get(kg_psi_modal),
          l_max);
      recorder.append_worldtube_mode_data(
          detail::dataset_name_for_component("/dtKGPsi"), time,
          get(kg_pi_modal), l_max);
    }
  }
}
}  // namespace Cce::TestHelpers
