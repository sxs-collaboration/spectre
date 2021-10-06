// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/*!
 * \ingroup DataStructuresGroup
 * \brief An untyped tensor component with a name for observation.
 *
 * The name should be a path inside an H5 file, typically starting with the name
 * of the volume subfile. For example,
 * `element_volume_data.vol/ObservationId[ID]/[ElementIdName]/psi_xx`.
 */
struct TensorComponent {
  TensorComponent() = default;
  TensorComponent(std::string in_name, DataVector in_data);
  TensorComponent(std::string in_name, std::vector<float> in_data);

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
  std::string name{};
  std::variant<DataVector, std::vector<float>> data{};
};

std::ostream& operator<<(std::ostream& os, const TensorComponent& t);

bool operator==(const TensorComponent& lhs, const TensorComponent& rhs);

bool operator!=(const TensorComponent& lhs, const TensorComponent& rhs);

/*!
 * \ingroup DataStructuresGroup
 * \brief Holds the extents of the mesh and the tensor components on the mesh.
 *
 * The extents is a `std::vector<size_t>` where each element is the number of
 * grid points in the given dimension. The `TensorComponent`s must live on the
 * grid of the size of the extents. We use runtime extents instead of the
 * `Index` class because observers may write 1D, 2D, or 3D data in a 3D
 * simulation.
 */
struct ExtentsAndTensorVolumeData {
  ExtentsAndTensorVolumeData() = default;
  ExtentsAndTensorVolumeData(std::vector<size_t> extents_in,
                             std::vector<TensorComponent> components);

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
  std::vector<size_t> extents{};
  std::vector<TensorComponent> tensor_components{};
};

/*!
 * An extension of `ExtentsAndTensorVolumeData` to store `Spectral::Quadrature`
 * and `Spectral::Basis`  associated with each axis of the element, in addition
 * to the extents and tensor components data.
 */
struct ElementVolumeData : ExtentsAndTensorVolumeData {
  ElementVolumeData() = default;
  ElementVolumeData(std::vector<size_t> extents_in,
                    std::vector<TensorComponent> components,
                    std::vector<Spectral::Basis> basis_in,
                    std::vector<Spectral::Quadrature> quadrature_in);

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
  std::vector<Spectral::Basis> basis{};
  std::vector<Spectral::Quadrature> quadrature{};
};
