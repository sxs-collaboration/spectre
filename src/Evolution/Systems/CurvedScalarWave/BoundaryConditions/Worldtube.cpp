// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/Worldtube.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "Utilities/Gsl.hpp"
namespace CurvedScalarWave::BoundaryConditions {

template <size_t Dim>
Worldtube<Dim>::Worldtube(CkMigrateMessage* const msg)
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Worldtube<Dim>::get_clone() const {
  return std::make_unique<Worldtube>(*this);
}

template <size_t Dim>
void Worldtube<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID Worldtube<Dim>::my_PUP_ID = 0;

template <size_t Dim>
std::optional<std::string> Worldtube<Dim>::dg_time_derivative(
    gsl::not_null<Scalar<DataVector>*> dt_psi_correction,
    gsl::not_null<Scalar<DataVector>*> dt_pi_correction,
    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> dt_phi_correction,
    const std::optional<tnsr::I<DataVector, Dim>>& face_mesh_velocity,
    const tnsr::i<DataVector, Dim>& normal_covector,
    const tnsr::I<DataVector, Dim>& normal_vector,
    const Scalar<DataVector>& /*psi*/, const Scalar<DataVector>& /*pi*/,
    const tnsr::i<DataVector, Dim>& phi, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim>& shift,
    const tnsr::II<DataVector, Dim,
                   Frame::Inertial>& /*inverse_spatial_metric*/,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const Scalar<DataVector>& /*dt_psi*/, const tnsr::i<DataVector, Dim>& d_psi,
    const tnsr::ij<DataVector, Dim>& d_phi,
    const Variables<tmpl::list<Tags::Psi, Tags::Pi,
                               Tags::Phi<Dim>>>& /*worldtube_vars*/) const {
  auto char_speeds =
      characteristic_speeds(gamma1, lapse, shift, normal_covector);
  if (face_mesh_velocity.has_value()) {
    const auto face_speed = dot_product(normal_covector, *face_mesh_velocity);
    for (size_t i = 0; i < 4; ++i) {
      char_speeds.at(i) -= get(face_speed);
    }
  }
  *dt_psi_correction =
      tenex::evaluate(normal_vector(ti::I) * (d_psi(ti::i) - phi(ti::i)));
  *dt_phi_correction = tenex::evaluate<ti::i>(
      normal_vector(ti::J) * (d_phi(ti::j, ti::i) - d_phi(ti::i, ti::j)));

  for (size_t i = 0; i < get(*dt_psi_correction).size(); ++i) {
    get(*dt_psi_correction)[i] *= std::min(0., gsl::at(char_speeds[0], i));
    for (size_t j = 0; j < Dim; ++j) {
      dt_phi_correction->get(j)[i] *= std::min(0., gsl::at(char_speeds[1], i));
    }
  }
  get(*dt_pi_correction) = get(gamma2) * get(*dt_psi_correction);
  return {};
}

template <size_t Dim>
std::optional<std::string> Worldtube<Dim>::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> psi,
    const gsl::not_null<Scalar<DataVector>*> pi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> phi,
    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> shift,
    const gsl::not_null<Scalar<DataVector>*> gamma1,
    const gsl::not_null<Scalar<DataVector>*> gamma2,
    const gsl::not_null<tnsr::II<DataVector, Dim, Frame::Inertial>*>
        inverse_spatial_metric,

    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
    /*face_mesh_velocity*/,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
    const Scalar<DataVector>& psi_interior,
    const Scalar<DataVector>& pi_interior,
    const tnsr::i<DataVector, Dim>& phi_interior,
    const Scalar<DataVector>& lapse_interior,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_interior,
    const tnsr::II<DataVector, Dim, Frame::Inertial>&
        inverse_spatial_metric_interior,
    const Scalar<DataVector>& gamma1_interior,
    const Scalar<DataVector>& gamma2_interior,
    const Scalar<DataVector>& /*dt_psi*/,
    const tnsr::i<DataVector, Dim>& /*d_psi*/,
    const tnsr::ij<DataVector, Dim>& /*d_phi*/,
    const Variables<tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>>>&
        worldtube_vars) const {
  ASSERT(worldtube_vars.number_of_grid_points() == get(lapse_interior).size(),
         "The worldtube solution has the wrong number of grid points: "
             << worldtube_vars.number_of_grid_points());
  *lapse = lapse_interior;
  *shift = shift_interior;
  *inverse_spatial_metric = inverse_spatial_metric_interior;
  *gamma1 = gamma1_interior;
  *gamma2 = gamma2_interior;

  const auto& psi_worldtube = get<Tags::Psi>(worldtube_vars);
  const auto& pi_worldtube = get<Tags::Pi>(worldtube_vars);
  const auto& phi_worldtube = get<Tags::Phi<Dim>>(worldtube_vars);

  // `VMinus` is calculated from the worldtube solution, the other
  // characteristic fields are set to the corresponding values of the evolved
  // variables. This is equivalent to giving no boundary conditions on those
  // fields.
  Variables<tmpl::list<Tags::VPsi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>>
      char_fields_mixed(get(gamma1_interior).size());

  auto& v_psi = get<Tags::VPsi>(char_fields_mixed);
  auto& v_zero = get<Tags::VZero<Dim>>(char_fields_mixed);
  auto& v_plus = get<Tags::VPlus>(char_fields_mixed);
  auto& v_minus = get<Tags::VMinus>(char_fields_mixed);

  // use allocation for temporary
  v_psi = dot_product(normal_vector, phi_interior);
  v_zero = tenex::evaluate<ti::i>(phi_interior(ti::i) -
                                  normal_covector(ti::i) * v_psi());
  v_plus = tenex::evaluate(pi_interior() + v_psi() -
                           gamma2_interior() * psi_interior());
  v_psi = psi_interior;
  v_minus = tenex::evaluate(pi_worldtube() -
                            normal_vector(ti::I) * phi_worldtube(ti::i) -
                            gamma2_interior() * psi_worldtube());

  evolved_fields_from_characteristic_fields(psi, pi, phi, gamma2_interior,
                                            v_psi, v_zero, v_plus, v_minus,
                                            normal_covector);
  return {};
}

template class Worldtube<3>;
}  // namespace CurvedScalarWave::BoundaryConditions
