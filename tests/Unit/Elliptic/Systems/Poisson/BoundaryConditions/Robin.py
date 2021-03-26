# Distributed under the MIT License.
# See LICENSE.txt for details.


def dirichlet_field(dirichlet_weight, constant, used_for_size):
    return constant / dirichlet_weight


def dirichlet_field_linearized(dirichlet_weight, constant, used_for_size):
    return 0.


def neumann_normal_dot_field_gradient(field, dirichlet_weight, neumann_weight,
                                      constant):
    return (constant - dirichlet_weight * field) / neumann_weight


def neumann_normal_dot_field_gradient_linearized(field_correction,
                                                 dirichlet_weight,
                                                 neumann_weight, constant):
    return -dirichlet_weight / neumann_weight * field_correction
