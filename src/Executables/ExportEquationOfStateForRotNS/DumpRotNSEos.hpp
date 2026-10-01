// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/Gsl.hpp"

namespace ExportEosForRotNS {

/// Interpolates `values_interp` at `target_log_density` from a table of
/// log10(rho) values. Uses cubic polynomial interpolation, falling back to
/// linear when values vary by more than `max_variation_for_linear_fallback`
/// across the stencil. Pass `std::numeric_limits<double>::max()` (the default)
/// to always use polynomial interpolation.
void interpolate_profile_at_log_density(
    gsl::not_null<double*> result, double target_log_density,
    const DataVector& log_rest_mass_density_interp,
    const DataVector& values_interp,
    double max_variation_for_linear_fallback =
        std::numeric_limits<double>::max());

/// Dumps a relativistic equation of state to disk in the format expected by
/// RotNS. The output has four columns: log10(n_cgs), log10(e_cgs), Y_e, and
/// log10(p_cgs).
///
/// \p thermodynamic_profile_filename, if provided, is a text file specifying
/// the thermodynamic profile as a function of rest-mass density (in geometric
/// units). Its first line must contain two integers: the number of entries and
/// a flag (0 or 1) indicating whether a Y_e(rho) column is present. Each
/// subsequent line has two columns (density, temperature) if the flag is 0, or
/// three columns (density, temperature, electron fraction) if the flag is 1.
/// The density range covered by this table must encompass the dump range given
/// by \p lower_bound_rest_mass_density_cgs and
/// \p upper_bound_rest_mass_density_cgs.
void dump_eos(const EquationsOfState::EquationOfState<true, 3>& eos,
              size_t number_of_log10_number_density_points_for_dump,
              const std::string& output_file_name,
              double lower_bound_rest_mass_density_cgs,
              double upper_bound_rest_mass_density_cgs,
              const std::optional<std::string>& thermodynamic_profile_filename);

}  // namespace ExportEosForRotNS
