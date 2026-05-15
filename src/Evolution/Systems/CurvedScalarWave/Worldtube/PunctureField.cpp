// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/PunctureField.hpp"

#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DynamicBuffer.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {

PunctureField::Schwarzschild::Schwarzschild(const size_t expansion_order_in,
                                            const double black_hole_mass_in,
                                            const Options::Context& /*context*/)
    : expansion_order(expansion_order_in),
      black_hole_mass(black_hole_mass_in) {}

PunctureField::Kerr::Kerr(const size_t expansion_order_in,
                          const double black_hole_mass_in,
                          const double spin_along_z_axis_in,
                          const Options::Context& /*context*/)
    : expansion_order(expansion_order_in),
      black_hole_mass(black_hole_mass_in),
      spin_along_z_axis(spin_along_z_axis_in) {}

PunctureField::PunctureField(const Schwarzschild& schwarzschild,
                             const Options::Context& /*context*/)
    : expansion_order_(schwarzschild.expansion_order),
      black_hole_mass_(schwarzschild.black_hole_mass) {}

PunctureField::PunctureField(const Kerr& kerr,
                             const Options::Context& /*context*/)
    : type_(Type::Kerr),
      expansion_order_(kerr.expansion_order),
      black_hole_mass_(kerr.black_hole_mass),
      spin_along_z_axis_(kerr.spin_along_z_axis) {}

void PunctureField::pup(PUP::er& p) {
  p | type_;
  p | expansion_order_;
  p | black_hole_mass_;
  p | spin_along_z_axis_;
}

PunctureField::Type PunctureField::type() const { return type_; }

size_t PunctureField::expansion_order() const { return expansion_order_; }

double PunctureField::black_hole_mass() const { return black_hole_mass_; }

double PunctureField::spin_along_z_axis() const { return spin_along_z_axis_; }

void PunctureField::apply_puncture(
    const gsl::not_null<Variables<tmpl::list<
        CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
        ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                      Frame::Inertial>>>*>
        result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& centered_coords,
    const tnsr::I<double, 3>& particle_position,
    const tnsr::I<double, 3>& particle_velocity,
    const tnsr::I<double, 3>& particle_acceleration) const {
  if (type_ == Type::Kerr) {
    ERROR("Kerr puncture not implemented yet");
  }
  if (expansion_order_ == 0) {
    puncture_field_0(result, centered_coords, particle_position,
                     particle_velocity, particle_acceleration,
                     black_hole_mass_);
  } else if (expansion_order_ == 1) {
    puncture_field_1(result, centered_coords, particle_position,
                     particle_velocity, particle_acceleration,
                     black_hole_mass_);
  } else {
    ERROR(
        "The puncture field is only implemented up to expansion order 1 but "
        "you requested order "
        << expansion_order_);
  }
}

void PunctureField::apply_acceleration_terms(
    const gsl::not_null<Variables<tmpl::list<
        CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
        ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                      Frame::Inertial>>>*>
        result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& centered_coords,
    const tnsr::I<double, 3>& particle_position,
    const tnsr::I<double, 3>& particle_velocity,
    const tnsr::I<double, 3>& particle_acceleration, const double ft,
    const double fx, const double fy, const double dt_ft, const double dt_fx,
    const double dt_fy, const double Du_ft, const double Du_fx,
    const double Du_fy, const double dt_Du_ft, const double dt_Du_fx,
    const double dt_Du_fy) const {
  if (type_ == Type::Kerr) {
    ERROR("Kerr puncture not implemented yet");
  }
  if (expansion_order_ == 0) {
    acceleration_terms_0(result, centered_coords, particle_position,
                         particle_velocity, particle_acceleration, ft, fx, fy,
                         dt_ft, dt_fx, dt_fy, black_hole_mass_);
  } else if (expansion_order_ == 1) {
    acceleration_terms_1(result, centered_coords, particle_position,
                         particle_velocity, particle_acceleration, ft, fx, fy,
                         dt_ft, dt_fx, dt_fy, Du_ft, Du_fx, Du_fy, dt_Du_ft,
                         dt_Du_fx, dt_Du_fy, black_hole_mass_);
  } else {
    ERROR(
        "The puncture field is only implemented up to expansion order 1 but "
        "you requested order "
        << expansion_order_);
  }
}

}  // namespace CurvedScalarWave::Worldtube
