// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/BoundaryConditions/ConstraintPreservingSphericalRadiation.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace ScalarWave::BoundaryConditions {
namespace detail {
ConstraintPreservingSphericalRadiationType
convert_constraint_preserving_spherical_radiation_type_from_yaml(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  if ("Sommerfeld" == type_read) {
    return ConstraintPreservingSphericalRadiationType::Sommerfeld;
  } else if ("FirstOrderBaylissTurkel" == type_read) {
    return ConstraintPreservingSphericalRadiationType::FirstOrderBaylissTurkel;
  } else if ("SecondOrderBaylissTurkel" == type_read) {
    return ConstraintPreservingSphericalRadiationType::SecondOrderBaylissTurkel;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read
                  << "\" to ConstraintPreservingSphericalRadiation::Type. Must "
                     "be one of Sommerfeld, FirstOrderBaylissTurkel, or "
                     "SecondOrderBaylissTurkel.");
}
}  // namespace detail

template <size_t Dim>
ConstraintPreservingSphericalRadiation<Dim>::
    ConstraintPreservingSphericalRadiation(
        const detail::ConstraintPreservingSphericalRadiationType type)
    : type_(type) {}

template <size_t Dim>
ConstraintPreservingSphericalRadiation<
    Dim>::ConstraintPreservingSphericalRadiation(CkMigrateMessage* const msg)
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
ConstraintPreservingSphericalRadiation<Dim>::get_clone() const {
  return std::make_unique<ConstraintPreservingSphericalRadiation>(*this);
}

template <size_t Dim>
void ConstraintPreservingSphericalRadiation<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
  p | type_;
}

template <size_t Dim>
std::optional<std::string>
ConstraintPreservingSphericalRadiation<Dim>::dg_time_derivative(
    const gsl::not_null<Scalar<DataVector>*> dt_pi_correction,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        dt_phi_correction,
    const gsl::not_null<Scalar<DataVector>*> dt_psi_correction,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const Scalar<DataVector>& psi,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    const Scalar<DataVector>& gamma2, const Scalar<DataVector>& dt_pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& dt_phi,
    const Scalar<DataVector>& dt_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_psi,
    const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_phi) const {
  {
    if (type_ ==
        detail::ConstraintPreservingSphericalRadiationType::Sommerfeld) {
      get(*dt_pi_correction) = -get(dt_pi);
      for (size_t i = 0; i < Dim; ++i) {
        get(*dt_pi_correction) += normal_covector.get(i) * dt_phi.get(i);
      }
    } else {
      DataVector& inv_radius = get(*dt_psi_correction);
      magnitude(dt_psi_correction, coords);
      inv_radius = 1.0 / inv_radius;

      if (type_ == detail::ConstraintPreservingSphericalRadiationType::
                       FirstOrderBaylissTurkel) {
        get(*dt_pi_correction) = -get(dt_pi) + inv_radius * get(dt_psi);
        for (size_t i = 0; i < Dim; ++i) {
          get(*dt_pi_correction) += normal_covector.get(i) * dt_phi.get(i);
        }
      } else {
        // second-order Bayliss-Turkel
        get(*dt_pi_correction) =
            -get(dt_pi) -
            2.0 * inv_radius * (2.0 * get(pi) - inv_radius * get(psi));
        for (size_t i = 0; i < Dim; ++i) {
          // Note: here we are handling `dt dr Psi` as `n^i dt Phi_i` and
          // `dr dt Psi` as `-n^i d_i Pi`. The only thing we are assuming is
          // that `d_r` is time-independent. This is why we have `n^i (dt Phi_i
          // - d_i Pi)` instead of `2 n^i dt Phi_i`.
          get(*dt_pi_correction) +=
              normal_covector.get(i) *
              (dt_phi.get(i) - d_pi.get(i) + 4.0 * inv_radius * phi.get(i));
          for (size_t j = 0; j < Dim; ++j) {
            get(*dt_pi_correction) += normal_covector.get(i) *
                                      normal_covector.get(j) * d_phi.get(i, j);
          }
        }
      }
    }
  }
  if (face_mesh_velocity.has_value()) {
    // Compute dt psi correction
    get(*dt_psi_correction) =
        get<0>(normal_covector) * (get<0>(d_psi) - get<0>(phi));
    for (size_t i = 1; i < Dim; ++i) {
      get(*dt_psi_correction) +=
          normal_covector.get(i) * (d_psi.get(i) - phi.get(i));
    }
    const auto negative_lambda0 =
        dot_product(normal_covector, *face_mesh_velocity);
    get(*dt_psi_correction) *= -get(negative_lambda0);
    get(*dt_pi_correction) += get(gamma2) * get(*dt_psi_correction);

    // Compute dt Phi 2-index constraint correction
    for (size_t i = 0; i < Dim; ++i) {
      dt_phi_correction->get(i) =
          get<0>(normal_covector) * (d_phi.get(0, i) - d_phi.get(i, 0));
      for (size_t j = 1; j < Dim; ++j) {
        dt_phi_correction->get(i) +=
            normal_covector.get(j) * (d_phi.get(j, i) - d_phi.get(i, j));
      }
      dt_phi_correction->get(i) *= -0.5 * get(negative_lambda0);
    }
    if (min(-get(negative_lambda0)) < 0.0) {
      return {
          "Incoming characteristic speeds for constraint preserving spherical "
          "radiation boundary condition. It's unclear that proper boundary "
          "conditions are imposed in this case. Please verify if you need this "
          "feature."};
    }
  } else {
    // Since the constraint-preserving terms are all multiplied by the
    // characteristic speeds, when those speeds are zero there are no
    // constraint-preserving terms to add.
    get(*dt_psi_correction) = 0.0;
    for (size_t i = 0; i < Dim; ++i) {
      dt_phi_correction->get(i) = 0.0;
    }
  }
  return {};
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID ConstraintPreservingSphericalRadiation<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class ConstraintPreservingSphericalRadiation<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace ScalarWave::BoundaryConditions
