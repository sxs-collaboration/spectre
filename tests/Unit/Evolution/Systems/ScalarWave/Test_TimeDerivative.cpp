// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Evolution/Systems/ScalarWave/TimeDerivative.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
auto add_scalar_to_tensor_components(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& input,
    double constant) noexcept {
  tnsr::i<DataVector, Dim, Frame::Inertial> copy_of_input =
      [&input, &constant]() noexcept {
        auto local_copy_of_input = input;
        for (size_t i = 0; i < Dim; ++i) {
          local_copy_of_input.get(i) += constant;
        }
        return local_copy_of_input;
      }();
  return copy_of_input;
}
template <size_t Dim>
void check_du_dt(const size_t npts, const double time) {
  ScalarWave::Solutions::PlaneWave<Dim> solution(
      make_array<Dim>(0.1), make_array<Dim>(0.0),
      std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(1.0, 1.0,
                                                                    0.0));

  tnsr::I<DataVector, Dim> x = [npts]() {
    auto logical_coords = logical_coordinates(Mesh<Dim>{
        3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto});
    tnsr::I<DataVector, Dim> coords{pow<Dim>(npts)};
    for (size_t i = 0; i < Dim; ++i) {
      coords.get(i) = std::move(logical_coords.get(i));
    }
    return coords;
  }();

  auto local_check_du_dt = [&npts, &time, &solution, &x](
                               const double gamma2, const double constraint) {
    auto box = db::create<db::AddSimpleTags<
        ScalarWave::Tags::ConstraintGamma2, Tags::dt<ScalarWave::Pi>,
        Tags::dt<ScalarWave::Phi<Dim>>, Tags::dt<ScalarWave::Psi>,
        ScalarWave::Pi, ScalarWave::Phi<Dim>,
        Tags::deriv<ScalarWave::Pi, tmpl::size_t<Dim>, Frame::Inertial>,
        Tags::deriv<ScalarWave::Psi, tmpl::size_t<Dim>, Frame::Inertial>,
        Tags::deriv<ScalarWave::Phi<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>>(
        Scalar<DataVector>(pow<Dim>(npts), gamma2),
        Scalar<DataVector>(pow<Dim>(npts), 0.0),
        tnsr::i<DataVector, Dim, Frame::Inertial>(pow<Dim>(npts), 0.0),
        Scalar<DataVector>(pow<Dim>(npts), 0.0),
        Scalar<DataVector>(-1.0 * solution.dpsi_dt(x, time).get()),
        add_scalar_to_tensor_components(solution.dpsi_dx(x, time), constraint),
        [&x, &time, &solution]() noexcept {
          auto dpi_dx = solution.d2psi_dtdx(x, time);
          for (size_t i = 0; i < Dim; ++i) {
            dpi_dx.get(i) *= -1.0;
          }
          return dpi_dx;
        }(),
        solution.dpsi_dx(x, time),
        [&npts, &x, &time, &solution]() noexcept {
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

    db::mutate_apply<
        tmpl::list<Tags::dt<ScalarWave::Pi>, Tags::dt<ScalarWave::Phi<Dim>>,
                   Tags::dt<ScalarWave::Psi>>,
        tmpl::push_front<
            typename ScalarWave::TimeDerivative<Dim>::argument_tags,
            Tags::deriv<ScalarWave::Pi, tmpl::size_t<Dim>, Frame::Inertial>,
            Tags::deriv<ScalarWave::Phi<Dim>, tmpl::size_t<Dim>,
                        Frame::Inertial>,
            Tags::deriv<ScalarWave::Psi, tmpl::size_t<Dim>, Frame::Inertial>>>(
        ScalarWave::TimeDerivative<Dim>{}, make_not_null(&box));

    CHECK_ITERABLE_APPROX(
        db::get<Tags::dt<ScalarWave::Pi>>(box),
        Scalar<DataVector>(-1.0 * solution.d2psi_dt2(x, time).get()));
    CHECK_ITERABLE_APPROX(
        db::get<Tags::dt<ScalarWave::Phi<Dim>>>(box),
        add_scalar_to_tensor_components(solution.d2psi_dtdx(x, time),
                                        -gamma2 * constraint));
    CHECK_ITERABLE_APPROX(db::get<Tags::dt<ScalarWave::Psi>>(box),
                          solution.dpsi_dt(x, time));
  };

  // Test with constraint damping parameter set to zero
  local_check_du_dt(0.0, 0.0);
  local_check_du_dt(0.0, 100.0);
  local_check_du_dt(0.0, -998.0);

  // Test with constraint satisfied but nonzero constraint damping parameter
  local_check_du_dt(10.0, 0.0);
  local_check_du_dt(-4.3, 0.0);

  // Test with one-index constraint NOT satisfied and nonzero constraint
  // damping parameter
  local_check_du_dt(10.0, 10.9);
  local_check_du_dt(1.2, -77.0);
  local_check_du_dt(-10.9, 43.0);
  local_check_du_dt(-90.0, -56.0);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarWave.TimeDerivative",
                  "[Unit][Evolution]") {
  constexpr double time = 0.7;
  check_du_dt<1>(3, time);
  check_du_dt<2>(3, time);
  check_du_dt<3>(3, time);
}
