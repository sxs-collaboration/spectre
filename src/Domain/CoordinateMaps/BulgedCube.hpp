// Distributed under the MIT License.
// See LICENSE.txt for details.

// Defines the class BulgedCube.

#pragma once

#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TypeTraits.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace domain {
namespace CoordinateMaps {

/*!
 *  \ingroup CoordinateMapsGroup
 *
 *  \brief Three dimensional map from the cube to a bulged cube.
 *  The cube is shaped such that the surface is compatible
 *  with the inner surface of Wedge3D.
 *  The shape of the object can be chosen to be cubical,
 *  if the sphericity is set to 0, or to a sphere, if
 *  the sphericity is set to 1. The sphericity can
 *  be set to any number between 0 and 1 for a bulged cube.
 *
 *  \details The volume map from the cube to a bulged cube is obtained by
 *  interpolating between six surface maps, twelve bounding curves, and
 *  eight corners. The surface map for the upper +z axis is obtained by
 *  interpolating between a cubical surface and a spherical surface. The
 *  two surfaces are chosen such that the latter circumscribes the former.
 *
 *  We make a choice here as to whether we wish to use the logical coordinates
 *  parameterizing these surface as they are, in which case we have the
 *  equidistant choice of coordinates, or whether to apply a tangent map to them
 *  which leads us to the equiangular choice of coordinates. In terms of the
 *  logical coordinates, the equiangular coordinates are:
 *
 *  \f[\textrm{equiangular xi} : \Xi(\xi) = \textrm{tan}(\xi\pi/4)\f]
 *
 *  \f[\textrm{equiangular eta}  : \mathrm{H}(\eta) = \textrm{tan}(\eta\pi/4)\f]
 *
 *  With derivatives:
 *
 *  \f[\Xi'(\xi) = \frac{\pi}{4}(1+\Xi^2)\f]
 *
 *  \f[\mathrm{H}'(\eta) = \frac{\pi}{4}(1+\mathrm{H}^2)\f]
 *
 *  The equidistant coordinates are:
 *
 *  \f[ \textrm{equidistant xi}  : \Xi = \xi\f]
 *
 *  \f[ \textrm{equidistant eta}  : \mathrm{H} = \eta\f]
 *
 *  with derivatives:
 *
 *  <center>\f$\Xi'(\xi) = 1\f$, and \f$\mathrm{H}'(\eta) = 1\f$</center>
 *
 *  We also define the variable \f$\rho\f$, given by:
 *
 *  \f[\rho = \sqrt{1+\Xi^2+\mathrm{H}^2}\f]
 *
 *  ### The Spherical Face Map
 *  The surface map for the spherical face of radius \f$R\f$ lying in the
 *  \f$+z\f$
 *  direction in either choice of coordinates is then given by:
 *
 *  \f[
 *  \vec{\sigma}_{spherical}(\xi,\eta) =
 *  \begin{bmatrix}
 *  x(\xi,\eta)\\
 *  y(\xi,\eta)\\
 *  z(\xi,\eta)\\
 *  \end{bmatrix}  = \frac{R}{\rho}
 *  \begin{bmatrix}
 *  \Xi\\
 *  \mathrm{H}\\
 *  1\\
 *  \end{bmatrix}
 *  \f]
 *
 *  ### The Cubical Face Map
 *  The surface map for the cubical face of side length \f$2L\f$ lying in the
 *  \f$+z\f$ direction is given by:
 *
 *  \f[
 *  \vec{\sigma}_{cubical}(\xi,\eta) =
 *  \begin{bmatrix}
 *  x(\xi,\eta)\\
 *  y(\xi,\eta)\\
 *  L\\
 *  \end{bmatrix}  = L
 *  \begin{bmatrix}
 *  \Xi\\
 *  \mathrm{H}\\
 *  1\\
 *  \end{bmatrix}
 *  \f]
 *
 *  ### The Bulged Face Map
 *  To construct the bulged map we interpolate between a cubical face map of
 *  side length \f$2L\f$ and a spherical face map of radius \f$R\f$, with the
 *  interpolation parameter being \f$s\f$, the `sphericity`.
 *  The surface map for the bulged face lying in the \f$+z\f$ direction is then
 *  given by:
 *
 *  \f[
 *  \vec{\sigma}_{+\zeta}(\xi,\eta) = \left\{(1-s)L + \frac{sR}{\rho}\right\}
 *  \begin{bmatrix}
 *  \Xi\\
 *  \mathrm{H}\\
 *  1\\
 *  \end{bmatrix}
 *  \f]
 *
 *  This equation defines the upper-z map \f$\vec{\sigma}_{+\zeta}\f$, and we
 *  similarly define the other five surface maps \f$\vec{\sigma}_{+\eta}\f$,
 *  \f$\vec{\sigma}_{+\xi}\f$, and so on by appropriate rotations.
 *  We constrain L by demanding that the spherical face circumscribe the cube.
 *  With this condition, we have \f$L = R/\sqrt3\f$.
 *
 *  ### The General Formula for 3D Isoparametric Maps
 *  The general formula is given by Eq. 1 in section 2.1 of Hesthaven's paper
 *  "A Stable Penalty Method For The Compressible Navier-Stokes Equations III.
 *  Multidimensional Domain Decomposition Schemes" available
 *  <a href="
 *  http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.699.1161&rep=rep1&type=pdf
 *  "> here </a>.
 *
 *  Hesthaven's formula is general in the degree of the shape functions used,
 *  so for our purposes we take the special case where the shape functions are
 *  linear in the interpolation variable, and define new variables accordingly.
 *  However, our interpolation variables do not necessarily have to be the
 *  logical coordinates themselves, though they often are. To make this
 *  distinction clear, we will define the new interpolation variables
 *  \f$\{\tilde{\xi},\tilde{\eta},\tilde{\zeta}\}\f$, which may either be the
 *  logical coordinates themselves or a invertible transformation of them. For
 *  the purposes of the bulged cube map, this transformation will be the same
 *  transformation that takes the logical coordinates into the equiangular
 *  coordinates. We will later see how this choice can lead to simplifications
 *  in the final map.
 *
 *  We define the following variables for
 *  \f$\alpha, \beta, \gamma \in\{\tilde{\xi},\tilde{\eta},\tilde{\zeta}\}\f$:
 *
 *  \f[
 *  f^{\pm}_{\alpha} = \frac{1}{2}(1\pm\alpha)\\
 *  f^{\pm\pm}_{\alpha \ \beta} = \frac{1}{4}(1\pm\alpha)(1\pm\beta)\\
 *  f^{\pm\pm\pm}_{\alpha \ \beta \ \gamma} =
 *  \frac{1}{8}(1\pm\alpha)(1\pm\beta)(1\pm\gamma)
 *  \f]
 *
 *  The formula involves six surfaces, which we will denote by
 *  \f$\vec{\sigma}\f$, twelve curves, denoted by \f$\vec{\Gamma}\f$, and eight
 *  vertices, denoted by \f$\vec{\pi}\f$, with the subscripts denoting which
 *  face(s) these objects belong to. The full volume map is given by:
 *
 *  \f{align*}
 *  \vec{x}(\xi,\eta,\zeta) = &
 *  f^{+}_{\tilde{\zeta}}\vec{\sigma}_{+\zeta}(\xi, \eta)+
 *  f^{-}_{\tilde{\zeta}}\vec{\sigma}_{-\zeta}(\xi, \eta)\\
 *  &+ f^{+}_{\tilde{\eta}}\vec{\sigma}_{+\eta}(\xi, \zeta)+
 *  f^{-}_{\tilde{\eta}}\vec{\sigma}_{-\eta}(\xi, \zeta)+
 *  f^{+}_{\tilde{\xi}}\vec{\sigma}_{+\xi}(\eta, \zeta)+
 *  f^{-}_{\tilde{\xi}}\vec{\sigma}_{-\xi}(\eta, \zeta)\\
 *  &- f^{++}_{\tilde{\xi} \ \tilde{\eta}}\vec{\Gamma}_{+\xi +\eta}(\zeta)-
 *  f^{-+}_{\tilde{\xi} \ \tilde{\eta}}\vec{\Gamma}_{-\xi +\eta}(\zeta)-
 *  f^{+-}_{\tilde{\xi} \ \tilde{\eta}}\vec{\Gamma}_{+\xi -\eta}(\zeta)-
 *  f^{--}_{\tilde{\xi} \ \tilde{\eta}}\vec{\Gamma}_{-\xi -\eta}(\zeta)\\
 *  &- f^{++}_{\tilde{\xi} \ \tilde{\zeta}}\vec{\Gamma}_{+\xi +\zeta}(\eta)-
 *  f^{-+}_{\tilde{\xi} \ \tilde{\zeta}}\vec{\Gamma}_{-\xi +\zeta}(\eta)-
 *  f^{+-}_{\tilde{\xi} \ \tilde{\zeta}}\vec{\Gamma}_{+\xi -\zeta}(\eta)-
 *  f^{--}_{\tilde{\xi} \ \tilde{\zeta}}\vec{\Gamma}_{-\xi -\zeta}(\eta)\\
 *  &- f^{++}_{\tilde{\eta} \ \tilde{\zeta}}\vec{\Gamma}_{+\eta +\zeta}(\xi)-
 *  f^{-+}_{\tilde{\eta} \ \tilde{\zeta}}\vec{\Gamma}_{-\eta +\zeta}(\xi)-
 *  f^{+-}_{\tilde{\eta} \ \tilde{\zeta}}\vec{\Gamma}_{+\eta -\zeta}(\xi)-
 *  f^{--}_{\tilde{\eta} \tilde{\zeta}}\vec{\Gamma}_{-\eta -\zeta}(\xi)\\
 *  &+ f^{+++}_{\tilde{\xi} \ \tilde{\eta} \ \tilde{\zeta}}\vec{\pi}_{+\xi +\eta
 * +\zeta}+ f^{-++}_{\tilde{\xi} \ \tilde{\eta} \ \tilde{\zeta}}\vec{\pi}_{-\xi
 * +\eta +\zeta}+ f^{+-+}_{\tilde{\xi} \ \tilde{\eta} \
 * \tilde{\zeta}}\vec{\pi}_{+\xi -\eta +\zeta}+
 *  f^{--+}_{\tilde{\xi} \ \tilde{\eta} \ \tilde{\zeta}}\vec{\pi}_{-\xi -\eta
 * +\zeta}\\
 *  &+ f^{++-}_{\tilde{\xi} \ \tilde{\eta} \ \tilde{\zeta}}\vec{\pi}_{+\xi +\eta
 * -\zeta}+ f^{-+-}_{\tilde{\xi} \ \tilde{\eta} \ \tilde{\zeta}}\vec{\pi}_{-\xi
 * +\eta -\zeta}+ f^{+--}_{\tilde{\xi} \ \tilde{\eta} \
 * \tilde{\zeta}}\vec{\pi}_{+\xi -\eta -\zeta}+ f^{---}_{\tilde{\xi} \
 * \tilde{\eta} \ \tilde{\zeta}}\vec{\pi}_{-\xi -\eta -\zeta} \f}
 *
 *
 *  ### The Special Case for Octahedral Symmetry
 *  The general formula is for the case in which there are six independently
 *  specified bounding surfaces. In our case, the surfaces are obtained by
 *  rotations and reflections of the upper-\f$\zeta\f$ face.
 *
 * We define the matrices corresponding to these transformations to be:
 *
 * \f[
 * S_{xy} =
 *  \begin{bmatrix}
 *  0 & 1 & 0\\
 *  1 & 0 & 0\\
 *  0 & 0 & 1\\
 *  \end{bmatrix},\
 *
 * S_{xz} =
 *  \begin{bmatrix}
 *  0 & 0 & 1\\
 *  0 & 1 & 0\\
 *  1 & 0 & 0\\
 *  \end{bmatrix},\
 *
 * S_{yz} =
 *  \begin{bmatrix}
 *  1 & 0 & 0\\
 *  0 & 0 & 1\\
 *  0 & 1 & 0\\
 *  \end{bmatrix}\f]
 *
 * \f[C_{zxy} =
 *  \begin{bmatrix}
 *  0 & 0 & 1\\
 *  1 & 0 & 0\\
 *  0 & 1 & 0\\
 *  \end{bmatrix},\
 *
 * C_{yzx} =
 *  \begin{bmatrix}
 *  0 & 1 & 0\\
 *  0 & 0 & 1\\
 *  1 & 0 & 0\\
 *  \end{bmatrix}\f]
 *
 * \f[N_{x} =
 *  \begin{bmatrix}
 *  -1 & 0 & 0\\
 *  0 & 1 & 0\\
 *  0 & 0 & 1\\
 *  \end{bmatrix},\
 *
 * N_{y} =
 *  \begin{bmatrix}
 *  1 & 0 & 0\\
 *  0 & -1 & 0\\
 *  0 & 0 & 1\\
 *  \end{bmatrix},\
 *
 * N_{z} =
 *  \begin{bmatrix}
 *  1 & 0 & 0\\
 *  0 & 1 & 0\\
 *  0 & 0 & -1\\
 *  \end{bmatrix}
 *  \f]
 *
 * The surface maps can now all be written in terms of
 * \f$\vec{\sigma}_{+\zeta}\f$ and these matrices:
 * <center>
 * \f$\vec{\sigma}_{-\zeta}(\xi, \eta) = N_z\vec{\sigma}_{+\zeta}(\xi, \eta)\\
 * \vec{\sigma}_{+\eta}(\xi, \zeta) = S_{yz}\vec{\sigma}_{+\zeta}(\xi, \zeta)\\
 * \vec{\sigma}_{-\eta}(\xi, \zeta) = N_yS_{yz}\vec{\sigma}_{+\zeta}(\xi,
 * \zeta)\\
 * \vec{\sigma}_{+\xi}(\eta, \zeta) = C_{zxy}\vec{\sigma}_{+\zeta}(\eta,
 * \zeta)\\
 * \vec{\sigma}_{-\xi}(\eta, \zeta) = N_xC_{zyx}\vec{\sigma}_{+\zeta}(\eta,
 * \zeta)\f$
 * </center>
 *
 * The four bounding curves \f$\vec{\Gamma}\f$ on the \f$+\zeta\f$ face are
 * given by:
 *
 * <center>
 * \f$\vec{\Gamma}_{+\xi,+\zeta}(\eta) = \vec{\sigma}_{+\zeta}(+1,\eta)\\
 * \vec{\Gamma}_{-\xi,+\zeta}(\eta) = \vec{\sigma}_{+\zeta}(-1,\eta)
 * = N_x\vec{\sigma}_{+\zeta}(+1, \eta)\\
 * \vec{\Gamma}_{+\eta,+\zeta}(\xi) = \vec{\sigma}_{+\zeta}(\xi,+1)
 * = S_{xy}\vec{\sigma}_{+\zeta}(+1, \xi)\\
 * \vec{\Gamma}_{-\eta,+\zeta}(\xi) = \vec{\sigma}_{+\zeta}(\xi,-1)
 * = N_yS_{xy}\vec{\sigma}_{+\zeta}(+1,\xi)\f$
 * </center>
 *
 * The bounding curves on the other surfaces can be obtained by transformations
 * on the \f$+\zeta\f$ face:
 *
 * <center>
 * \f$\vec{\Gamma}_{+\xi,-\zeta}(\eta) = N_z\vec{\sigma}_{+\zeta}(+1,\eta)\\
 * \vec{\Gamma}_{-\xi,-\zeta}(\eta) = N_z\vec{\sigma}_{+\zeta}(-1,\eta)
 * = N_zN_x\vec{\sigma}_{+\zeta}(+1,\eta)\\
 * \vec{\Gamma}_{+\eta,-\zeta}(\xi) = N_z\vec{\sigma}_{+\zeta}(\xi,+1)
 * = N_zS_{xy}\vec{\sigma}_{+\zeta}(+1, \xi)\\
 * \vec{\Gamma}_{-\eta,-\zeta}(\xi) = N_z\vec{\sigma}_{+\zeta}(\xi,-1)
 * = N_zN_yS_{xy}\vec{\sigma}_{+\zeta}(+1, \xi)\\
 * \vec{\Gamma}_{+\xi,+\eta}(\zeta) =
 * C_{zxy}\vec{\sigma}_{+\zeta}(+1,\zeta)\\
 * \vec{\Gamma}_{-\xi,+\eta}(\zeta) =
 * N_xC_{zxy}\vec{\sigma}_{+\zeta}(+1,\zeta)\\
 * \vec{\Gamma}_{+\xi,-\eta}(\zeta) = C_{zxy}\vec{\sigma}_{+\zeta}(-1,\zeta)
 * = C_{zxy}N_x\vec{\sigma}_{+\zeta}(+1,\zeta)\\
 * \vec{\Gamma}_{-\xi,-\eta}(\zeta) = N_xC_{zxy}\vec{\sigma}_{+\zeta}(-1,\zeta)
 * = N_xC_{zxy}N_x\vec{\sigma}_{+\zeta}(+1,\zeta)\f$
 * </center>
 *
 * Now we can write the volume map in terms of
 * \f$\vec{\sigma}_{+\zeta}\f$ only:
 * \f{align*}\vec{x}(\xi,\eta,\zeta) = &
 * (f^{+}_{\tilde{\zeta}} + f^{-}_{\tilde{\zeta}}N_z)
 * \vec{\sigma}_{+\zeta}(\xi, \eta)\\
 * &+ (f^{+}_{\tilde{\eta}} + f^{-}_{\tilde{\eta}}N_y)
 * S_{yz}\vec{\sigma}_{+\zeta}(\xi, \zeta)\\
 * &+ (f^{+}_{\tilde{\xi}} + f^{-}_{\tilde{\xi}}N_x)
 * C_{zxy}\vec{\sigma}_{+\zeta}(\eta, \zeta)\\
 * &- (f^{+}_{\tilde{\xi}}+f^{-}_{\tilde{\xi}}N_x)
 * (f^{+}_{\tilde{\eta}}+f^{-}_{\tilde{\eta}}N_y)
 * C_{zxy}\vec{\sigma}_{+\zeta}(+1, \zeta)\\
 * &- (f^{+}_{\tilde{\zeta}}+f^{-}_{\tilde{\zeta}}N_z)\left\{
 * (f^{+}_{\tilde{\xi}}+f^{-}_{\tilde{\xi}}N_x)\vec{\sigma}_{+\zeta}(+1, \eta)+
 * (f^{+}_{\tilde{\eta}}+f^{-}_{\tilde{\eta}}N_y)S_{xy}\vec{\sigma}_{+\zeta}(+1,
 * \xi)\right\}\\
 * &+ \frac{r}{\sqrt{3}}\vec{\tilde{\xi}}
 *  \f}
 *
 * Note that we can now absorb all of the \f$f\f$s into the matrix prefactors
 * in the above equation and obtain a final set of matrices. We define the
 * following *blending matrices*:
 *
 *  \f[
 *  B_{\tilde{\xi}} =
 *  \begin{bmatrix}
 *  0 & 0 & \tilde{\xi}\\
 *  1 & 0 & 0\\
 *  0 & 1 & 0\\
 *  \end{bmatrix},\
 *
 *  B_{\tilde{\eta}} =
 *  \begin{bmatrix}
 *  1 & 0 & 0\\
 *  0 & 0 & \tilde{\eta}\\
 *  0 & 1 & 0\\
 *  \end{bmatrix},\
 *
 *  B_{\tilde{\zeta}} =
 *  \begin{bmatrix}
 *  1 & 0 & 0\\
 *  0 & 1 & 0\\
 *  0 & 0 & \tilde{\zeta}\\
 *  \end{bmatrix}\\
 *
 *  B_{\tilde{\xi}\tilde{\eta}} =
 *  \begin{bmatrix}
 *  0 & 0 & \tilde{\xi}\\
 *  \tilde{\eta} & 0 & 0\\
 *  0 & 1 & 0\\
 *  \end{bmatrix},\
 *
 *  B_{\tilde{\xi}\tilde{\zeta}} =
 *  \begin{bmatrix}
 *  \tilde{\xi} & 0 & 0\\
 *  0 & 1 & 0\\
 *  0 & 0 & \tilde{\zeta}\\
 *  \end{bmatrix},\
 *
 *  B_{\tilde{\eta}\tilde{\zeta}} =
 *  \begin{bmatrix}
 *  0 & 1 & 0\\
 *  \tilde{\eta} & 0 & 0\\
 *  0 & 0 & \tilde{\zeta}\\
 *  \end{bmatrix}\\
 *
 *  B_{\tilde{\xi}\tilde{\eta}\tilde{\zeta}} =
 *  \begin{bmatrix}
 *  \tilde{\xi} & 0 & 0\\
 *  0 & \tilde{\eta} & 0\\
 *  0 & 0 & \tilde{\zeta}\\
 *  \end{bmatrix}
 *  \f]
 *
 *  Now we can write the volume map in these terms:
 *
 * \f{align*}
 * \vec{x}(\xi,\eta,\zeta) = &
 * B_{\tilde{\zeta}}
 * \vec{\sigma}_{+\zeta}(\xi, \eta)\\& +
 * B_{\tilde{\eta}}
 * \vec{\sigma}_{+\zeta}(\xi, \zeta)+
 * B_{\tilde{\xi}}
 * \vec{\sigma}_{+\zeta}(\eta, \zeta)\\& -
 * B_{\tilde{\xi} \tilde{\eta}}
 * \vec{\sigma}_{+\zeta}(+1, \zeta)-
 * B_{\tilde{\xi} \tilde{\zeta}}
 * \vec{\sigma}_{+\zeta}(+1, \eta)+
 * B_{\tilde{\eta} \tilde{\zeta}}
 * \vec{\sigma}_{+\zeta}(+1, \xi)\\& +
 * B_{\tilde{\xi} \tilde{\eta} \tilde{\zeta}}
 * \vec{\sigma}_{+\zeta}(+1, +1)
 * \f}
 *
 * ### The Bulged Cube Map
 * We now use the result above to provide the mapping for the bulged cube.
 * First we will define the variables \f$\rho_A\f$ and \f$\rho_{AB}\f$, for
 * \f$A, B \in \{\Xi,\mathrm{H}, \mathrm{Z}\} \f$, where \f$\mathrm{Z}\f$
 * is \f$\tan(\zeta\pi/4)\f$ in the equiangular case and \f$\zeta\f$ in the
 * equidistant case:
 *
 * \f[
 * \rho_A = \sqrt{2 + A^2}\\
 * \rho_{AB} = \sqrt{1 + A^2 + B^2}
 * \f]
 * The final mapping is then:
 * \f[
 * \vec{x}(\xi,\eta,\zeta) = \frac{(1-s)R}{\sqrt{3}}
 * \begin{bmatrix}
 * \Xi\\
 * \mathrm{H}\\
 * \mathrm{Z}\\
 * \end{bmatrix} +
 * \frac{sR}{\sqrt{3}}
 * \begin{bmatrix}
 * \tilde{\xi}\\
 * \tilde{\eta}\\
 * \tilde{\zeta}\\
 * \end{bmatrix} + sR
 * \begin{bmatrix}
 * \tilde{\xi} & \Xi & \Xi\\
 * \mathrm{H} & \tilde{\eta} &\mathrm{H}\\
 * \mathrm{Z} & \mathrm{Z} & \tilde{\zeta}\\
 * \end{bmatrix}
 * \begin{bmatrix}
 * 1/\rho_{\mathrm{H}\mathrm{Z}}\\
 * 1/\rho_{\Xi\mathrm{Z}}\\
 * 1/\rho_{\Xi\mathrm{H}}\\
 * \end{bmatrix} - sR
 * \begin{bmatrix}
 * \Xi & \tilde{\xi} & \tilde{\xi}\\
 * \tilde{\eta} & \mathrm{H} &\tilde{\eta}\\
 * \tilde{\zeta} & \tilde{\zeta} & \mathrm{Z}\\
 * \end{bmatrix}
 * \begin{bmatrix}
 * 1/\rho_{\Xi}\\
 * 1/\rho_{\mathrm{H}}\\
 * 1/\rho_{\mathrm{Z}}\\
 * \end{bmatrix}
 * \f]
 *
 * Recall that the lower case Greek letters with tildes are the variables
 * used for the linear interpolation between the six bounding surfaces, and
 * that the upper case Greek letters are the coordinates along these surfaces -
 * both of which can be specified to be either
 * equidistant or equiangular. In the case where the
 * interpolation variable is chosen to match that of the
 * coordinates along the surface, we have \f$\tilde{\xi} = \Xi\f$, etc. In this
 * case, the formula reduces further. The reduced formula below is the one used
 * for this CoordinateMap. It is given by:
 *
 * \f[
 * \vec{x}(\xi,\eta,\zeta) =
 * \left\{
 * \frac{R}{\sqrt{3}}
 * + sR
 * \left(
 * 1/\rho_{\mathrm{H}\mathrm{Z}}+
 * 1/\rho_{\Xi\mathrm{Z}}+
 * 1/\rho_{\Xi\mathrm{H}}-
 * 1/\rho_{\Xi}-
 * 1/\rho_{\mathrm{H}}-
 * 1/\rho_{\mathrm{Z}}
 * \right)
 * \right\}
 * \begin{bmatrix}
 * \Xi\\
 * \mathrm{H}\\
 * \mathrm{Z}\\
 * \end{bmatrix}
 * \f]
 *
 * The inverse mapping is analytic in the angular directions. A root find
 * must be performed for the inverse mapping in the radial direction. This
 * one-dimensional formula is obtained by taking the magnitude of both sides
 * of the mapping, and changing variables from \f$\xi, \eta, \zeta\f$ to
 * \f$x, y, z\f$ and introducing \f$\rho^2 := \sqrt{\xi^2+\eta^2+\zeta^2}\f$.
 */
class BulgedCube {
 public:
  static constexpr size_t dim = 3;
  BulgedCube(double radius, double sphericity,
             bool use_equiangular_map) noexcept;
  BulgedCube() noexcept = default;
  ~BulgedCube() noexcept = default;
  BulgedCube(BulgedCube&&) noexcept = default;
  BulgedCube(const BulgedCube&) noexcept = default;
  BulgedCube& operator=(const BulgedCube&) noexcept = default;
  BulgedCube& operator=(BulgedCube&&) noexcept = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const noexcept;

  boost::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords) const noexcept;

  // clang-tidy: google runtime references
  void pup(PUP::er& p) noexcept;  // NOLINT

  bool is_identity() const noexcept { return is_identity_; }

 private:
  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> xi_derivative(
      const std::array<T, 3>& source_coords) const noexcept;
  friend bool operator==(const BulgedCube& lhs, const BulgedCube& rhs) noexcept;

  double radius_{std::numeric_limits<double>::signaling_NaN()};
  double sphericity_{std::numeric_limits<double>::signaling_NaN()};
  bool use_equiangular_map_ = false;
  bool is_identity_ = false;
};

bool operator!=(const BulgedCube& lhs, const BulgedCube& rhs) noexcept;
}  // namespace CoordinateMaps
}  // namespace domain
