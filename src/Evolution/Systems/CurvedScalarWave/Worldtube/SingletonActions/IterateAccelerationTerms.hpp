// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>

#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "NumericalAlgorithms/Strahlkorper/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {

/*!
 * \brief Computes the next iteration of the acceleration due to scalar self
 * force from the current iteration of the regular field, as well as the
 * quantities required to compute the acceleration terms of the puncture field.
 *
 * \details Analytic expressions for the computed terms are given in Section V.B
 * of \cite Wittek:2024gxn.
 */
struct IterateAccelerationTerms {
  static constexpr size_t Dim = 3;
  using simple_tags = tmpl::list<Tags::AccelerationTerms>;
  using return_tags = tmpl::list<Tags::AccelerationTerms>;
  using argument_tags = tmpl::list<
      Tags::ParticlePositionVelocity<Dim>, Tags::BackgroundQuantities<Dim>,
      Tags::GeodesicAcceleration<Dim>,
      Stf::Tags::StfTensor<Tags::PsiWorldtube, 0, Dim, Frame::Inertial>,
      Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 0, Dim,
                           Frame::Inertial>,
      Stf::Tags::StfTensor<Tags::PsiWorldtube, 1, Dim, Frame::Inertial>,
      Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 1, Dim,
                           Frame::Inertial>,
      Tags::Charge, Tags::Mass, ::Tags::Time, Tags::SelfForceTurnOnTime,
      Tags::SelfForceTurnOnInterval, Tags::CurrentIteration>;
  static void apply(
      gsl::not_null<Scalar<DataVector>*> acceleration_terms,
      const std::array<tnsr::I<double, Dim>, 2>& pos_vel,
      const tuples::TaggedTuple<
          gr::Tags::SpacetimeMetric<double, Dim>,
          gr::Tags::InverseSpacetimeMetric<double, Dim>,
          gr::Tags::SpacetimeChristoffelSecondKind<double, Dim>,
          gr::Tags::TraceSpacetimeChristoffelSecondKind<double, Dim>,
          Tags::TimeDilationFactor>& background,
      const tnsr::I<double, Dim, Frame::Inertial>& geodesic_acc,
      const Scalar<double>& psi_monopole, const Scalar<double>& dt_psi_monopole,
      const tnsr::i<double, Dim, Frame::Inertial>& psi_dipole,
      const tnsr::i<double, Dim, Frame::Inertial>& dt_psi_dipole, double charge,
      std::optional<double> mass, double time,
      std::optional<double> turn_on_time,
      std::optional<double> turn_on_interval, size_t iteration);
};

}  // namespace CurvedScalarWave::Worldtube
