// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/AnalyticConstant.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryConditions/ConstraintPreserving.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryConditions/DirichletAnalytic.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarTensor::BoundaryConditions {
/// Typelist of standard BoundaryConditions.
using standard_boundary_conditions =
    tmpl::list<ScalarTensor::BoundaryConditions::ConstraintPreserving,
               ScalarTensor::BoundaryConditions::DemandOutgoingCharSpeeds,
               ScalarTensor::BoundaryConditions::DirichletAnalytic,
               domain::BoundaryConditions::Periodic<BoundaryCondition>>;

}  // namespace ScalarTensor::BoundaryConditions
