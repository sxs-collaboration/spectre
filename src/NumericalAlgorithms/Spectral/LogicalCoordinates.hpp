// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions logical_coordinates and interface_logical_coordinates

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace Tags {
template<size_t Dim>
struct Mesh;
template <size_t, typename>
struct Coordinates;
}  // namespace Tags
}  // namespace domain

template <size_t Dim>
class Mesh;
class DataVector;
template <size_t Dim>
class Direction;

namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

/// @{
/*!
 * \ingroup ComputationalDomainGroup
 * \brief Compute the logical coordinates of a Mesh in an Element.
 *
 * \details The logical coordinates are the collocation points
 * associated with the Spectral::Basis and Spectral::Quadrature of the \p mesh.
 * The Spectral::Basis determines the domain of the logical coordinates, while
 * the Spectral::Quadrature determines their distribution.  For Legendre or
 * Chebyshev bases, the logical coordinates are in the interval \f$[-1, 1]\f$.
 * These bases may have either GaussLobatto or Gauss quadrature, which are not
 * uniformly distributed, and either include (GaussLobatto) or do not include
 * (Gauss) the end points.  For the FiniteDifference basis, the logical
 * coordinates are again in the interval \f$[-1, 1]\f$.  This basis may have
 * either FaceCentered or CellCentered quadrature, which are uniformly
 * distributed, and either include (FaceCentered) or do not include
 * (CellCentered) the end points. The Fourier basis has logical coordinates
 * in the interval \f$[0, 2 \pi]\f$. The SphericalHarmonic basis corresponds to
 * the spherical coordinates \f$(\theta, \phi)\f$ where the polar angle
 * \f$\theta\f$ is in the interval \f$[0, \pi]\f$ and the azimuth \f$\phi\f$ is
 * in the interval \f$[0, 2 \pi]\f$.  The polar angle has Gauss quadrature
 * corresponding to the Legendre Gauss points of \f$\cos \theta\f$ (and thus
 * have no points at the poles), while the azimuth has Equiangular quadrature
 * which are distributed uniformly including the left endpoint, but not the
 * right. For the Cartoon basis, only one logical coordinate is allowed and it
 * is never used for computations. The Zernike bases have the radial basis
 * implemented such that the logical coordinates are \f$[-1, 1]\f$, while
 * an affine map is internally used to go to \f$[0, 1]\f$, where these bases are
 * defined. Being a scaled one-sided Jacobi polynomial, the gridpoint are
 * shifted to upper \f$r\f$, and GaussRadauUpper quadrature has been implemented
 * such that the lower edge does not have an end point but the upper does.
 *
 * \example
 * \snippet Test_LogicalCoordinates.cpp logical_coordinates_example
 */
template <size_t VolumeDim>
void logical_coordinates(
    gsl::not_null<tnsr::I<DataVector, VolumeDim, Frame::ElementLogical>*>
        logical_coords,
    const Mesh<VolumeDim>& mesh);

template <size_t VolumeDim>
tnsr::I<DataVector, VolumeDim, Frame::ElementLogical> logical_coordinates(
    const Mesh<VolumeDim>& mesh);
/// @}

namespace domain {
namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The logical coordinates in the Element
template <size_t VolumeDim>
struct LogicalCoordinates : Coordinates<VolumeDim, Frame::ElementLogical>,
                            db::ComputeTag {
  using base = Coordinates<VolumeDim, Frame::ElementLogical>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<Mesh<VolumeDim>>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<return_type*>, const ::Mesh<VolumeDim>&)>(
      &logical_coordinates<VolumeDim>);
};
}  // namespace Tags
}  // namespace domain
