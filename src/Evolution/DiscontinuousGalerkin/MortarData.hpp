// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <optional>
#include <pup.h>
#include <string>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace evolution::dg {
/*!
 * \brief Data on the mortar used to compute the boundary correction for the
 * DG scheme.
 *
 * The class holds the data that has been projected to one side of the mortar.
 * It is meant to be used in a container (either MortarDataHolder or
 * TimeSteppers::BoundaryHistory) that holds MortarData on each side of the
 * mortar. The data is later used to compute the same unique boundary correction
 * on the mortar for both elements. That is, the final boundary correction
 * computation is done twice: once on each element touching the mortar. However,
 * the computation is done in such a way that the results agree.
 *
 * For local time stepping, the magnitude of the face normal is stored.
 *
 * The magnitude of the face normal is given by:
 *
 * \f{align*}{
 *  \sqrt{
 *   \frac{\partial\xi}{\partial x^i} \gamma^{ij}
 *   \frac{\partial\xi}{\partial x^j}}
 * \f}
 *
 * for a face in the \f$\xi\f$-direction, with inverse spatial metric
 * \f$\gamma^{ij}\f$.
 *
 * In addition, for local time stepping with Gauss points, the determinants of
 * the volume inverse Jacobian and the face Jacobian are stored.
 *
 * In addition to the (type-erased) fields on the mortar, the face mesh is
 * stored
 *
 * If the element and its neighbor have unaligned logical coordinate
 * systems then the data and meshes are stored in the local logical
 * coordinate's orientation (\f$\xi\f$ varies fastest). This means the
 * action sending the data is responsible for reorienting the data on
 * the mortar so it matches the neighbor's orientation.
 *
 * \tparam Dim the volume dimension
 */
template <size_t Dim>
struct MortarData {
  std::optional<DataVector> mortar_data{std::nullopt};
  std::optional<Scalar<DataVector>> face_normal_magnitude{std::nullopt};
  std::optional<Scalar<DataVector>> face_det_jacobian{std::nullopt};
  std::optional<Scalar<DataVector>> volume_det_inv_jacobian{std::nullopt};
  std::optional<Mesh<Dim - 1>> face_mesh{std::nullopt};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
};

template <size_t Dim>
bool operator==(const MortarData<Dim>& lhs, const MortarData<Dim>& rhs);

template <size_t Dim>
bool operator!=(const MortarData<Dim>& lhs, const MortarData<Dim>& rhs);

template <size_t Dim>
std::ostream& operator<<(std::ostream& os, const MortarData<Dim>& mortar_data);
}  // namespace evolution::dg
