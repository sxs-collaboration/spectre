// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"

namespace ExportEosForRotNS {

/// Dumps a relativistic equation of state to disk in the format expected by
/// RotNS. The output has four columns: log10(n_cgs), log10(e_cgs), Y_e, and
/// log10(p_cgs).
///
/// If the equation of state is not barotropic, the temperature defaults to the
/// EoS lower bound and the electron fraction is set to beta equilibrium.
void dump_eos(const EquationsOfState::EquationOfState<true, 3>& eos,
              size_t number_of_log10_number_density_points_for_dump,
              const std::string& output_file_name,
              double lower_bound_rest_mass_density_cgs,
              double upper_bound_rest_mass_density_cgs);

}  // namespace ExportEosForRotNS
