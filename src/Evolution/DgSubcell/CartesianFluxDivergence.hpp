// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Index;
template <size_t Dim, typename Frame>
class ElementMap;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
namespace Frame {
struct Inertial;
struct Grid;
}  // namespace Frame
namespace domain {
template <typename SourceFrame, typename TargetFrame, size_t Dim>
class CoordinateMapBase;
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
}  // namespace domain
/// \endcond

namespace evolution::dg::subcell {
/// @{
/*!
 * \brief Compute and add the 2nd-order flux divergence on a Cartesian mesh to
 * the cell-centered time derivatives.
 */
void add_cartesian_flux_divergence(gsl::not_null<DataVector*> dt_var,
                                   double one_over_delta,
                                   const DataVector& inv_jacobian,
                                   const DataVector& boundary_correction,
                                   const Index<1>& subcell_extents,
                                   size_t dimension);

void add_cartesian_flux_divergence(gsl::not_null<DataVector*> dt_var,
                                   double one_over_delta,
                                   const DataVector& inv_jacobian,
                                   const DataVector& boundary_correction,
                                   const Index<2>& subcell_extents,
                                   size_t dimension);

void add_cartesian_flux_divergence(gsl::not_null<DataVector*> dt_var,
                                   double one_over_delta,
                                   const DataVector& inv_jacobian,
                                   const DataVector& boundary_correction,
                                   const Index<3>& subcell_extents,
                                   size_t dimension);
/// @}

/*!
 * \brief Compute and add the 2nd-order flux divergence on a Cartesian mesh to
 * the cell-centered time derivatives when some of the bases are Cartoon.
 *
 * \details Symmetries of your spacetime, used in the cartoon method, allow
 * the replacement of derivatives perpendicular to the computational domain by
 * scaled components of that same tensor. Specifically, for a spherically
 * symmetric system, this allows
 * \f[
 * \partial_t u + \partial_x F^x + \partial_y F^y + \partial_z F^z \Rightarrow
 * \partial_t u + \frac{1}{x^2} \partial_x \left( x^2 F^x \right),
 * \f]
 * and for an axially symmetric system,
 * \f[
 * \partial_t u + \partial_x F^x + \partial_y F^y + \partial_z F^z \Rightarrow
 * \partial_t u + \frac{1}{x} \partial_x \left( x F^x \right) + \partial_y F^y.
 * \f]
 *
 * \note This function will assume your basis is Cartoon by `subcell_extents ==
 * 1` in that dimension (only allowed in the second or third dimension).
 */
void add_cartoon_cartesian_flux_divergence(
    gsl::not_null<DataVector*> dt_var, double one_over_delta,
    const DataVector& inv_jacobian, const DataVector& boundary_correction,
    const Index<3>& subcell_extents, size_t dimension,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const ElementMap<3, Frame::Grid>& logical_to_grid_map,
    const ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial_map,
    double time,
    const std::unordered_map<
    std::string, std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time);
}  // namespace evolution::dg::subcell
