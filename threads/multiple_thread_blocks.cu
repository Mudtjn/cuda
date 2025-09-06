#include<bits/stdc++.h>
using namespace std; 

#define N 100

__global__ 
void MatAdd( float A[N][N], float B[N][N], float C[N][N]) {
    
} 

int main() {

    float A[N][N], B[N][N], C[N][N];
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            A[i][j] = i+j;
            B[i][j] = i-j;
        }
    }

    
}