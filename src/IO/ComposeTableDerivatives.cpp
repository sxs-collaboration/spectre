// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/ComposeTableDerivatives.hpp"

#include <cmath>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace io {
namespace {

size_t idx_of(const size_t in, const size_t iT, const size_t iYe,
              const size_t nN, const size_t nYe) {
  return (iT * nN + in) * nYe + iYe;
}

}  // namespace

DataVector compute_zeta_from_free_energy_derivatives(
    const DataVector& d2f_dt2, const DataVector& d2f_dt_dnb,
    const DataVector& d2f_dt_dye, const DataVector& d2f_dnb_dye,
    const DataVector& df_dye, const std::vector<double>& number_density_grid,
    const std::vector<double>& temperature_grid,
    const size_t number_density_points, const size_t temperature_points,
    const size_t electron_fraction_points) {
  // Local aliases for the densely-used grid sizes.
  const size_t nN = number_density_points;
  const size_t nT = temperature_points;
  const size_t nYe = electron_fraction_points;
  const size_t ntot = nN * nT * nYe;
  ASSERT(d2f_dt2.size() == ntot,
         "d2F/dT2 size " << d2f_dt2.size() << " does not match table size "
                         << ntot << ".");
  ASSERT(d2f_dt_dnb.size() == ntot,
         "d2F/dTdn_b size " << d2f_dt_dnb.size()
                            << " does not match table size " << ntot << ".");
  ASSERT(d2f_dt_dye.size() == ntot,
         "d2F/dTdY_e size " << d2f_dt_dye.size()
                            << " does not match table size " << ntot << ".");
  ASSERT(d2f_dnb_dye.size() == ntot,
         "d2F/dn_bdY_e size " << d2f_dnb_dye.size()
                              << " does not match table size " << ntot << ".");
  ASSERT(df_dye.size() == ntot, "dF/dY_e size " << df_dye.size()
                                                << " does not match table size "
                                                << ntot << ".");
  ASSERT(number_density_grid.size() == nN,
         "Number-density grid size " << number_density_grid.size()
                                     << " does not match nN " << nN << ".");
  ASSERT(temperature_grid.size() == nT,
         "Temperature grid size " << temperature_grid.size()
                                  << " does not match nT " << nT << ".");

  DataVector zeta(ntot);
  // eps_T = -T d2F/dT2
  // This floor only guards against unphysical/degenerate table entries (e.g. T
  // = 0).
  constexpr double eps_T_floor = 1e-300;

  for (size_t in = 0; in < nN; ++in) {
    const double nb = number_density_grid[in];
    const double nb_squared = nb * nb;
    for (size_t iT = 0; iT < nT; ++iT) {
      const double temperature = temperature_grid[iT];
      for (size_t iYe = 0; iYe < nYe; ++iYe) {
        const size_t idx = idx_of(in, iT, iYe, nN, nYe);

        // epsilon = F - T F_T, so its derivatives are
        //   eps_Ye = F_Ye - T F_{T,Ye},  eps_T = -T F_{T,T}.
        const double eps_Ye = df_dye[idx] - temperature * d2f_dt_dye[idx];
        const double eps_T = -temperature * d2f_dt2[idx];

        if (std::abs(eps_T) < eps_T_floor) {
          ERROR(
              "Encountered eps_T = -T d2F/dT2 ~ 0 while computing zeta at idx="
              << idx << "; cannot hold epsilon fixed when changing Y_e.");
        }

        // pressure p = n_b^2 F_{n_b}, so
        //   p_Ye = n_b^2 F_{n_b,Ye},  p_T = n_b^2 F_{n_b,T}.
        const double p_Ye = nb_squared * d2f_dnb_dye[idx];
        const double p_T = nb_squared * d2f_dt_dnb[idx];

        zeta[idx] = p_Ye - p_T * eps_Ye / eps_T;
      }
    }
  }

  return zeta;
}

}  // namespace io
