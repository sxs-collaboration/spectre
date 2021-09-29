// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/Burgers/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"  // IWYU pragma: keep

template Variables<tmpl::list<
    Tags::div<Tags::Flux<Burgers::Tags::U, tmpl::size_t<1>, Frame::Inertial>>>>
divergence(
    const Variables<tmpl::list<
        Tags::Flux<Burgers::Tags::U, tmpl::size_t<1>, Frame::Inertial>>>& F,
    const Mesh<1>& mesh,
    const InverseJacobian<DataVector, 1, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian) noexcept;
