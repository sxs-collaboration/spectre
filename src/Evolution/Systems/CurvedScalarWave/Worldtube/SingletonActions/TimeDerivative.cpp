
// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/TimeDerivative.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave::Worldtube {

void TimeDerivativeMutator::apply(
    const gsl::not_null<Variables<
        tmpl::list<::Tags::dt<Tags::Psi0>, ::Tags::dt<Tags::dtPsi0>>>*>
        dt_evolved_vars,
    const Variables<tmpl::list<Tags::Psi0, Tags::dtPsi0>>& evolved_vars,
    const Scalar<double>& psi_monopole,
    const tnsr::i<double, Dim, Frame::Grid>& psi_dipole,
    const tnsr::ii<double, Dim, Frame::Grid>& psi_quadrupole,
    const tnsr::i<double, Dim, Frame::Grid>& dt_psi_dipole,
    const tnsr::AA<double, Dim, Frame::Grid>& inverse_spacetime_metric,
    const tnsr::A<double, Dim, Frame::Grid>& trace_spacetime_christoffel,
    const ExcisionSphere<Dim>& excision_sphere) {
  const double wt_radius = excision_sphere.radius();
  const auto& psi0 = get(get<Tags::Psi0>(evolved_vars));
  const auto& dt_psi0 = get(get<Tags::dtPsi0>(evolved_vars));
  get(get<::Tags::dt<Tags::Psi0>>(*dt_evolved_vars)) = dt_psi0;
  double trace_inverse_spatial_metric = 0.;
  auto& dt2_psi0 = get(get<::Tags::dt<Tags::dtPsi0>>(*dt_evolved_vars));
  dt2_psi0 = get<0>(trace_spacetime_christoffel) * dt_psi0;
  for (size_t i = 0; i < Dim; ++i) {
    dt2_psi0 -=
        2. * inverse_spacetime_metric.get(0, i + 1) * dt_psi_dipole.get(i);
    dt2_psi0 += trace_spacetime_christoffel.get(i + 1) * psi_dipole.get(i);
    trace_inverse_spatial_metric += inverse_spacetime_metric.get(i + 1, i + 1);
    for (size_t j = 0; j < 3; ++j) {
      dt2_psi0 -= 2. * inverse_spacetime_metric.get(i + 1, j + 1) *
                  psi_quadrupole.get(i, j);
    }
  }
  dt2_psi0 -= 2. * trace_inverse_spatial_metric * (get(psi_monopole) - psi0) /
              (wt_radius * wt_radius);
  dt2_psi0 /= inverse_spacetime_metric.get(0, 0);
}

}  // namespace CurvedScalarWave::Worldtube
