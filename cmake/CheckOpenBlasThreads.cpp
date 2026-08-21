// Distributed under the MIT License.
// See LICENSE.txt for details
extern "C" {
void openblas_set_num_threads(int);
}

int main() { openblas_set_num_threads(1); }
