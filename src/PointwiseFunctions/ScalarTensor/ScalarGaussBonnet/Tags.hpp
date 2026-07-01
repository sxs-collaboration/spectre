// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarTensor/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/CouplingParameters.hpp"

/// \cond
namespace ScalarTensor::OptionTags {
struct Group;
}  // namespace ScalarTensor::OptionTags
/// \endcond

namespace ScalarTensor {
namespace OptionTags {
/*!
 * \brief Linear coupling parameters to curvature.
 */
struct CouplingParameters {
  static constexpr Options::String help = {"Coupling parameters to curvature."};
  using type = ScalarTensor::CouplingParameterOptions;
  using group = ScalarTensor::OptionTags::Group;
};

}  // namespace OptionTags

namespace Tags {
/*!
 * \brief Linear, quadratic and quartic coupling parameters to curvature.
 */
struct CouplingParameters : db::SimpleTag {
  using type = ScalarTensor::CouplingParameterOptions;
  using option_tags = tmpl::list<OptionTags::CouplingParameters>;
  static constexpr bool pass_metavariables = false;
  static ScalarTensor::CouplingParameterOptions create_from_options(
      const ScalarTensor::CouplingParameterOptions& coupling_parameters) {
    return coupling_parameters;
  }
};
}  // namespace Tags

namespace sgb::Tags {
/*!
 * \brief Double normal projection of the second covariant derivative of the
 * coupling function.
 *
 * \details Tag for the term $n^a n^b \nabla_a\nabla_b F[\Psi]$, where $F[\Psi]$
 * is the coupling function, and $n^a$ is the unit vector normal to the spatial
 * hypersurfaces.
 */
struct nnDDCoupling : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief Mixed projection of the second covariant derivative of the coupling
 * function.
 *
 * \details Tag for the term $\gamma^a_i n^b \nabla_a \nabla_b F[\Psi]$, where
 * $F[\Psi]$ is the coupling function, $n^a$ is the unit vector normal to the
 * spatial hypersurfaces, and $\gamma^a_b = \delta^a_b + n^a n_b$ is the
 * projection operator onto them.
 */
struct nsDDCoupling : db::SimpleTag {
  using type = tnsr::i<DataVector, 3>;
};

/*!
 * \brief Spatial projection of the second covariant derivative of the coupling
 * function.
 *
 * \details Tag for the term $\gamma^a_i \gamma^b_j \nabla_a \nabla_b F[\Psi]$,
 * where $F[\Psi]$ is the coupling function, $\gamma^a_b = \delta^a_b + n^a n_b$
 * is the projection operator onto the spatial hypersurfaces, and $n^a$ is the
 * unit vector normal to them.
 */
struct ssDDCoupling : db::SimpleTag {
  using type = tnsr::ii<DataVector, 3>;
};

/*!
 * \brief Spatial trace of the second covariant derivative of the coupling
 * function.
 *
 * \details Tag for the term $\gamma^{ab} \nabla_a \nabla_b F[\Psi]$, where
 * $F[\Psi]$ is the coupling function and $\gamma^{ab}$ is the inverse spatial
 * metric.
 */
struct SpatialTraceDDCoupling : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace sgb::Tags
}  // namespace ScalarTensor
