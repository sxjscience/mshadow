// only need tensor.h
#include "../mshadow/tensor.h"
#include "../mshadow/tensor_container.h"
#include <cstdio>
// simple file to test if it compiles
using namespace mshadow;
using namespace mshadow::expr;

void print(const CTensor2D &t) {
    index_t row = t.shape[1];
    index_t col = t.shape[0];
    printf("%2d X %2d\n", row, col);
    for (index_t r = 0; r < row; ++r) {
        for (index_t c = 0; c < col; ++c) {
            printf("%.2f ", t[r][c]);
        }
        printf("\n");
    }
}
// implemented by testcuda.cu
void testmkl( CTensor2D mat1, CTensor2D mat2, CTensor2D mat3 );

int main( void ){
    TensorContainer<cpu,2> lhs( Shape2(4,3), 0 );
    TensorContainer<cpu,2> rhs( Shape2(4,3), 0 );
    TensorContainer<cpu,2> dst( Shape2(4,4), 0.1 );
    lhs = 1.0f;
    print(lhs);
    printf("-\n");
    rhs[0] = 2.0f;
    rhs[1] = 0.0f;
    rhs[2] = 3.0f;

    print(rhs);
    // A += 0.1*dot(B.T(),C)
    //dst += 0.1 * dot(lhs.T(), rhs);
    dst -= 0.1 *dot( lhs, rhs.T() );
    print(dst);

    return 0;
}