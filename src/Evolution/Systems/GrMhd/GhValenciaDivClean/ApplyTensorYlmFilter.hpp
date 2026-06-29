// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/TensorYlm/ApplyFilter.hpp"
#include "NumericalAlgorithms/TensorYlm/Filter.hpp"

/// \cond
class DataVector;
namespace RadiationTransport::NoNeutrinos {
struct System;
}  // namespace RadiationTransport::NoNeutrinos
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
template <typename TagsList>
class Variables;
/// \endcond

namespace grmhd::GhValenciaDivClean::filter_detail {

/// The combined GH + Valencia variables tag list for GhValenciaDivClean.
using ghmhd_vars_list = typename grmhd::GhValenciaDivClean::System<
    RadiationTransport::NoNeutrinos::System>::variables_tag::tags_list;

/// The Valencia variables in their "grid frame" (used for YLM filtering).
/// Scalars are frame-independent; TildeS uses Grid frame lower index;
/// TildeB uses Grid frame upper index.
using valencia_grid_vars_list =
    tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
               grmhd::ValenciaDivClean::Tags::TildeYe,
               grmhd::ValenciaDivClean::Tags::TildeTau,
               grmhd::ValenciaDivClean::Tags::TildeS<Frame::Grid>,
               grmhd::ValenciaDivClean::Tags::TildeB<Frame::Grid>,
               grmhd::ValenciaDivClean::Tags::TildePhi>;

}  // namespace grmhd::GhValenciaDivClean::filter_detail

namespace ylm::TensorYlm {

/*!
 * \brief Applies TensorYlm filter to the combined GH+Valencia variables.
 *
 * Applies the GH specialization to the GH part of the variables and
 * applies a standard scalar/vector filter to the Valencia variables.
 */
template <>
void apply_tensor_ylm_filter(
    gsl::not_null<
        Variables<grmhd::GhValenciaDivClean::filter_detail::ghmhd_vars_list>*>
        vars,
    gsl::not_null<
        Variables<grmhd::GhValenciaDivClean::filter_detail::ghmhd_vars_list>*>
        temp_storage,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
        jac_grid_to_inertial,
    const ylm::TensorYlm::FilterMatrixHolder& filter_matrices, size_t ell_max,
    size_t radial_extents);

}  // namespace ylm::TensorYlm
