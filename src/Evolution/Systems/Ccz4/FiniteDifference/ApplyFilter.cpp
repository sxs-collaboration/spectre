// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/ApplyFilter.hpp"

#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Filter.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::fd {

void ApplyFilter::apply(
    const gsl::not_null<typename System::variables_tag::type*> evolved_vars_ptr,
    const Mesh<3>& subcell_mesh, const bool evolve_lapse_and_shift,
    const double kreiss_oliger_epsilon,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>& ghost_data) {
  constexpr size_t fd_order = 4;

  if (kreiss_oliger_epsilon == 0.0) {
    return;
  } else if (kreiss_oliger_epsilon > 0.0 and kreiss_oliger_epsilon <= 1.0) {
    typename System::variables_tag::type filtered_vars = *evolved_vars_ptr;

    Ccz4::fd::ccz4_kreiss_oliger_filter(make_not_null(&filtered_vars),
                                        *evolved_vars_ptr, ghost_data,
                                        evolve_lapse_and_shift, subcell_mesh,
                                        fd_order + 2, kreiss_oliger_epsilon);

    *evolved_vars_ptr = filtered_vars;
  } else {
    ERROR("Kreiss-Oliger epsilon should be in the interval [0, 1]. Got "
          << kreiss_oliger_epsilon << " instead.");
  }
}

}  // namespace Ccz4::fd
