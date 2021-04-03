// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/BoundaryConditions/SphericalRadiation.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarWave::BoundaryConditions {
namespace detail {
SphericalRadiationType convert_spherical_radiation_type_from_yaml(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  if ("Sommerfeld" == type_read) {
    return SphericalRadiationType::Sommerfeld;
  } else if ("BaylissTurkel" == type_read) {
    return SphericalRadiationType::BaylissTurkel;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read
                  << "\" to SphericalRadiation::Type. Must be one of "
                     "Sommerfeld, or BaylissTurkel.");
}
}  // namespace detail

template <size_t Dim>
SphericalRadiation<Dim>::SphericalRadiation(
    const detail::SphericalRadiationType type) noexcept
    : type_(type) {}

template <size_t Dim>
SphericalRadiation<Dim>::SphericalRadiation(
    CkMigrateMessage* const msg) noexcept
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
SphericalRadiation<Dim>::get_clone() const noexcept {
  return std::make_unique<SphericalRadiation>(*this);
}

template <size_t Dim>
void SphericalRadiation<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
  p | type_;
}

template <size_t Dim>
std::optional<std::string> SphericalRadiation<Dim>::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> pi_ext,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> phi_ext,
    const gsl::not_null<Scalar<DataVector>*> psi_ext,
    const gsl::not_null<Scalar<DataVector>*> gamma2_ext,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const Scalar<DataVector>& psi,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    const Scalar<DataVector>& gamma2) const noexcept {
  *gamma2_ext = gamma2;
  get(*pi_ext) = get<0>(normal_covector) * get<0>(phi);
  for (size_t i = 1; i < Dim; ++i) {
    get(*pi_ext) += normal_covector.get(i) * phi.get(i);
  }

  if (type_ == detail::SphericalRadiationType::BaylissTurkel) {
    // computed radius, storing int psi_ext
    DataVector& radius = get(*psi_ext);
    radius = square(get<0>(coords));
    for (size_t i = 1; i < Dim; ++i) {
      radius += square(coords.get(i));
    }
    radius = sqrt(radius);
    get(*pi_ext) += get(psi) / radius;
  }

  // Spherical radiation on a spherical boundary means no changes in Phi and
  // Psi. These would ideally be controlled by a constraint-preserving boundary
  // condition.
  *phi_ext = phi;
  get(*psi_ext) = get(psi);

  if (face_mesh_velocity.has_value()) {
    const Scalar<DataVector> char_speed =
        dot_product(*face_mesh_velocity, normal_covector);
    if (min(-get(char_speed)) < 0.0) {
      return {
          "Incoming characteristic speeds for spherical radiation boundary "
          "condition. It's unclear that proper boundary conditions are imposed "
          "in this case. Please verify if you need this feature."};
    }
  }
  return {};
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID SphericalRadiation<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template class SphericalRadiation<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace ScalarWave::BoundaryConditions
