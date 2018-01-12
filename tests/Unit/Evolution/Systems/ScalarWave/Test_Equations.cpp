// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Evolution/Systems/ScalarWave/Equations.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
template <size_t Dim>
void check_du_dt(const size_t npts, const double time) {
  ScalarWave::Solutions::PlaneWave<Dim> solution(
      make_array<Dim>(0.1), make_array<Dim>(0.0),
      std::make_unique<MathFunctions::Gaussian>(1.0, 1.0, 0.0));

  tnsr::I<DataVector, Dim> x = [npts]() {
    auto logical_coords = logical_coordinates(Index<Dim>(3));
    tnsr::I<DataVector, Dim> coords{pow<Dim>(npts)};
    for (size_t i = 0; i < Dim; ++i) {
      coords.get(i) = std::move(logical_coords.get(i));
    }
    return coords;
  }();

  auto box = db::create<db::AddTags<
      Tags::dt<ScalarWave::Pi>, Tags::dt<ScalarWave::Phi<Dim>>,
      Tags::dt<ScalarWave::Psi>, ScalarWave::Pi,
      Tags::deriv<ScalarWave::Pi, tmpl::size_t<Dim>, Frame::Inertial>,
      Tags::deriv<ScalarWave::Phi<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>>(
      Scalar<DataVector>(pow<Dim>(npts), 0.0),
      tnsr::i<DataVector, Dim, Frame::Inertial>(pow<Dim>(npts), 0.0),
      Scalar<DataVector>(pow<Dim>(npts), 0.0),
      Scalar<DataVector>(-1.0 * solution.dpsi_dt(x, time).get()),
      [&x, &time, &solution ]() noexcept {
        auto dpi_dx = solution.d2psi_dtdx(x, time);
        for (size_t i = 0; i < Dim; ++i) {
          dpi_dx.get(i) *= -1.0;
        }
        return dpi_dx;
      }(),
      [&npts, &x, &time, &solution ]() noexcept {
        tnsr::ij<DataVector, Dim, Frame::Inertial> d2psi_dxdx{pow<Dim>(npts),
                                                              0.0};
        const tnsr::ii<DataVector, Dim, Frame::Inertial> ddpsi_soln =
            solution.d2psi_dxdx(x, time);
        for (size_t i = 0; i < Dim; ++i) {
          for (size_t j = 0; j < Dim; ++j) {
            d2psi_dxdx.get(i, j) = ddpsi_soln.get(i, j);
          }
        }
        return d2psi_dxdx;
      }());

  db::mutate_apply<typename ScalarWave::ComputeDuDt<Dim>::return_tags,
                   typename ScalarWave::ComputeDuDt<Dim>::argument_tags>(
      ScalarWave::ComputeDuDt<Dim>{}, box);

  CHECK_ITERABLE_APPROX(
      db::get<Tags::dt<ScalarWave::Pi>>(box),
      Scalar<DataVector>(-1.0 * solution.d2psi_dt2(x, time).get()));
  CHECK_ITERABLE_APPROX(db::get<Tags::dt<ScalarWave::Phi<Dim>>>(box),
                        solution.d2psi_dtdx(x, time));
  CHECK_ITERABLE_APPROX(db::get<Tags::dt<ScalarWave::Psi>>(box),
                        solution.dpsi_dt(x, time));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarWave.DuDt",
                  "[Unit][Evolution]") {
  constexpr double time = 0.7;
  check_du_dt<1>(3, time);
  check_du_dt<2>(3, time);
  check_du_dt<3>(3, time);
}
