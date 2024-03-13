// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace CurvedScalarWave::Worldtube {

/*!
 * \brief Computes the final acceleration of the particle at this time step.
 * \details If `max_iterations` is 0, the acceleration will simply be
 * geodesic, see `gr::geodesic_acceleration`.  Otherwise, the acceleration due
 * to the scalar self-force is additionally applied to it, see
 * `self_force_acceleration`. This mutator is run on the worldtube
 * singleton chare.
 */
struct UpdateAcceleration {
  static constexpr size_t Dim = 3;
  using variables_tag = ::Tags::Variables<
      tmpl::list<Tags::EvolvedPosition<Dim>, Tags::EvolvedVelocity<Dim>>>;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  using return_tags = tmpl::list<dt_variables_tag>;
  using argument_tags = tmpl::list<
      Tags::ParticlePositionVelocity<Dim>, Tags::BackgroundQuantities<Dim>,
      Tags::GeodesicAcceleration<Dim>,
      Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 0, Dim,
                           Frame::Inertial>,
      Stf::Tags::StfTensor<Tags::PsiWorldtube, 1, Dim, Frame::Inertial>,
      Tags::Charge, Tags::Mass, Tags::MaxIterations>;
  static void apply(
      gsl::not_null<
          Variables<tmpl::list<::Tags::dt<Tags::EvolvedPosition<Dim>>,
                               ::Tags::dt<Tags::EvolvedVelocity<Dim>>>>*>
          dt_evolved_vars,
      const std::array<tnsr::I<double, Dim>, 2>& pos_vel,
      const tuples::TaggedTuple<gr::Tags::SpacetimeMetric<double, Dim>,
                                gr::Tags::InverseSpacetimeMetric<double, Dim>,
                                Tags::TimeDilationFactor>& background,
      const tnsr::I<double, Dim, Frame::Inertial>& geodesic_acc,
      const Scalar<double>& dt_psi_monopole,
      const tnsr::i<double, Dim, Frame::Inertial>& psi_dipole, double charge,
      std::optional<double> mass, size_t max_iterations);
};

}  // namespace CurvedScalarWave::Worldtube
