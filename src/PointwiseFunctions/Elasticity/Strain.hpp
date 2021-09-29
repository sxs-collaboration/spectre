// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace Elasticity {

/*!
 * \brief The symmetric strain \f$S_{ij} = \partial_{(i} \xi_{j)}\f$ on a flat
 * background in Cartesian coordinates.
 */
template <typename DataType, size_t Dim>
void strain(gsl::not_null<tnsr::ii<DataType, Dim>*> strain,
            const tnsr::iJ<DataType, Dim>& deriv_displacement) noexcept;

/*!
 * \brief The symmetric strain \f$S_{ij} = \nabla_{(i} \gamma_{j)k} \xi^k =
 * \partial_{(i} \gamma_{j)k} \xi^k - \Gamma_{kij} \xi^k\f$ on a
 * background metric \f$\gamma_{ij}\f$.
 */
template <typename DataType, size_t Dim>
void strain(gsl::not_null<tnsr::ii<DataType, Dim>*> strain,
            const tnsr::iJ<DataType, Dim>& deriv_displacement,
            const tnsr::ii<DataType, Dim>& metric,
            const tnsr::ijj<DataType, Dim>& deriv_metric,
            const tnsr::ijj<DataType, Dim>& christoffel_first_kind,
            const tnsr::I<DataType, Dim>& displacement) noexcept;

/*!
 * \brief The symmetric strain \f$S_{ij} = \partial_{(i} \xi_{j)}\f$ on a flat
 * background in Cartesian coordinates.
 *
 * Note that this function involves a numeric differentiation of the
 * displacement vector.
 */
template <size_t Dim>
void strain(gsl::not_null<tnsr::ii<DataVector, Dim>*> strain,
            const tnsr::I<DataVector, Dim>& displacement, const Mesh<Dim>& mesh,
            const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                                  Frame::Inertial>& inv_jacobian) noexcept;

namespace Tags {

/*!
 * \brief The symmetric strain \f$S_{ij}=\nabla_{(i} \xi_{j)}\f$ in the elastic
 * material.
 *
 * \see `Elasticity::strain`
 */
template <size_t Dim>
struct StrainCompute : Elasticity::Tags::Strain<Dim>, db::ComputeTag {
  using base = Elasticity::Tags::Strain<Dim>;
  using return_type = tnsr::ii<DataVector, Dim>;
  using argument_tags =
      tmpl::list<Elasticity::Tags::Displacement<Dim>, domain::Tags::Mesh<Dim>,
                 domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                               Frame::Inertial>>;
  static constexpr auto function = &strain<Dim>;
};

}  // namespace Tags
}  // namespace Elasticity
