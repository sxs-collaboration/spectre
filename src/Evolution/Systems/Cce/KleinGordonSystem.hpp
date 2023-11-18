// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {

/*!
 * \brief Performing Cauchy characteristic evolution and Cauchy characteristic
 * matching for Einstein-Klein-Gordon system.
 *
 * \details The code adopts the characteristic formulation to solve the field
 * equations for scalar-tensor theory as considered in \cite Ma2023sok. Working
 * in the Einstein frame, a real-valued scalar field \f$\psi\f$ is minimally
 * coupled with the spacetime metric \f$g_{\mu\nu}\f$. The corresponding action
 * is expressed as follows:
 *
 * \f[
 * S = \int d^4x \sqrt{-g} \left(\frac{R}{16 \pi}  - \frac{1}{2} \nabla_\mu \psi
 * \nabla^\mu \psi\right).
 * \f]
 *
 * The system consists of two sectors: scalar and tensor (metric). The scalar
 * field follows the Klein-Gordon (KG) equation
 *
 * \f[
 * \Box \psi = 0.
 * \f]
 *
 * Its characteristic expression is given in \cite Barreto2004fn, yielding
 * the hypersurface equation for \f$\partial_u\psi=\Pi\f$, where
 * \f$\partial_u\f$ represents differentiation with respect to retarded time $u$
 * at fixed numerical radius \f$y\f$. The code first integrates the KG equation
 * radially to determine \f$\Pi\f$. Subsequently, the time integration is
 * performed to evolve the scalar field \f$\psi\f$ forward in time.
 *
 * The tensor (metric) sector closely aligns with the current GR CCE system,
 * incorporating additional source terms that depend only on the scalar field
 * \f$\psi\f$ and its spatial derivatives, rather than its time derivative
 * \f$(\Pi)\f$. This feature preserves the hierarchical structure of the
 * equations. As a result, the Einstein-Klein-Gordon system can be divided into
 * three major sequential steps:
 *  - Integrate the metric hypersurface equations with the existing
 *    infrastructure
 *  - Integrate the KG equation for \f$\Pi\f$
 *  - Evolve two variables, \f$\psi\f$ (scalar) and \f$J\f$ (tensor) to the next
 *    time step
 *
 */
template <bool EvolveCcm>
struct KleinGordonSystem {
  static constexpr size_t volume_dim = 3;
  using variables_tag = tmpl::list<
      ::Tags::Variables<tmpl::list<Tags::BondiJ, Tags::KleinGordonPsi>>,
      ::Tags::Variables<tmpl::conditional_t<
          EvolveCcm,
          tmpl::list<Cce::Tags::CauchyCartesianCoords,
                     Cce::Tags::PartiallyFlatCartesianCoords,
                     Cce::Tags::InertialRetardedTime>,
          tmpl::list<Cce::Tags::CauchyCartesianCoords,
                     Cce::Tags::InertialRetardedTime>>>>;

  static constexpr bool has_primitive_and_conservative_vars = false;
};
} // namespace Cce
