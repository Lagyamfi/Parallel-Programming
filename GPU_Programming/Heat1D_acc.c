#include <stdio.h>
#include <stdlib.h>


// Define here as constant for easy change
#define REAL float

void printV (REAL A[], int N)
{
  int x;
  for (x=0; x<=N+1; x++)
  {
    printf("%d: %1.10f\n", x, A[x]);
  }
}

int main(int argc, char **argv)
{
  int  x, t, N= 10000000, T=100;
  REAL L= 0.001234, L2, S;
  REAL *U1, *U2;

  if (argc>1) { T = atoi(argv[1]); } // get  first command line parameter
  if (argc>2) { N = atoi(argv[2]); } // get second command line parameter
  if (argc>3) { L = atof(argv[3]); } // get  third command line parameter
 
  if (N < 1 || T < 1 || L >= 0.5) {
    printf("arguments: T N L (T: steps, N: vector size, L < 0.5)\n");
    return 1;
  }

  U1 = (REAL *) malloc ( sizeof(REAL)*(N+2) );
  U2 = (REAL *) malloc ( sizeof(REAL)*(N+2) );
  if (!U1 || !U2) { printf("Cannot allocate vectors\n"); exit(1); }

  // initialize temperatures at time t=0  
  for (x=0; x<=N+1; x++)
    U1[x] = U2[x] = 0.0;
 
  // initialize fixed boundary conditions on U1 and U2
  {
    U1[0]  = U2[0]  = 1.0;
    U1[N+1]= U2[N+1]= 9.0;
  }

  // Set initial temperature on two special points
  {
    U1[N/3]  = 8.0;
    U1[4*N/7]= 3.0;
  }

  printf("Stencil computation of %d steps on 1-D vector of %d elements with L=%1.10e\n", T, N, L);

  L2 = (1.0f-2.0f*L);
  // We need to care about the copying of the U1 and U2 vectors to the GPU
  // However this should only be done once(one time copyin, one time copyout)
  // the acc data copy command provides exactly this
  #pragma acc data copy(U1[:N+2], U2[:N+2])
  {
  // its is important to note that it is not possible to parallelize the time loop
  for (t=1; t<=T; t++)  // loop on time
    if ((t&1) == 1) // t is odd
    {
	  // We leave the details to the compiler by only commanding acc kernels loop
	  // Since we know that there is no data dependency in this inner loop, we add 
	  // the independent key word
      #pragma acc kernels loop independent
      for (x=1; x<=N; x++)  // loop on 1D bar
        U2[x] = L2*U1[x] + L*U1[x+1] + L*U1[x-1];
	}
    else            // t is even
    {
      #pragma acc kernels loop independent
      for (x=1; x<=N; x++)  // loop on 1D bar
        U1[x] = L2*U2[x] + L*U2[x+1] + L*U2[x-1];
	}
  }

  //exit(0);

  if ((t&1) == 1)
  {
    if (N<=100)
      printV(U1,N);
    else
    {
	  // we first thought about putting acc commands also to the sum loop
	  // of the check sum, but concluded that this would only have a neglectable effect 
	  //#pragma acc kernels loop independent
      for (S=0.0, x=0; x<=N+1; x++) // compute checksum of final state 
         S = S + U1[x];
      printf("\nCheckSum = %1.10e\n", S);
    }
  } 
  else
  {
    if (N<=100)
      printV(U2,N);
    else
    {
	  //#pragma acc kernels loop independent
      for (S=0.0, x=0; x<=N+1; x++) // compute checksum of final state 
         S = S + U2[x];
      printf("\nCheckSum = %1.10e\n", S);
    }
  }
  return 0;
}
s