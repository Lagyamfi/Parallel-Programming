//Lawrence Asamoah Adu-Gyamfi (1484610), Maximilian Grunwald (1519529)
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

//we used the provided code of the laplace application and turned this into
//a parallelized version by using OpenMP commands

float stencil ( float v1, float v2, float v3, float v4)
{
  return (v1 + v2 + v3 + v4) * 0.25f;
}

float max_error ( float prev_error, float old, float new )
{
  float t= fabsf( new - old );
  return t>prev_error? t: prev_error;
}

float laplace_step(float *in, float *out, int n)
{
  int i, j;
  float error=0.0f;
	#pragma omp for nowait//here the matrix gets subdivided according to the number of threads set; each thread still runs the
	for ( j=1; j < n-1; j++ )// number n of iterations, however only a fraction of the matrix is used each
    for ( i=1; i < n-1; i++ )// nowait allows the threads to continue calculations without a barrier
    {
      out[j*n+i]= stencil(in[j*n+i+1], in[j*n+i-1], in[(j-1)*n+i], in[(j+1)*n+i]);
      error = max_error( error, out[j*n+i], in[j*n+i] );
    }
  return error;
}

// we also tried a parallelization in the laplace_init. However, we couldn't perceive
// differnces in the runtime, due to the small amount of work that can be saved here, 
// compared with the double for-loop in the laplace_step
void laplace_init(float *in, int n)
{
  int i;
  const float pi  = 2.0f * asinf(1.0f);
  memset(in, 0, n*n*sizeof(float));

  for (i=0; i<n; i++) {
    float V = in[i*n] = sinf(pi*i / (n-1));
    in[ i*n+n-1 ] = V*expf(-pi);
  }
}

int main(int argc, char** argv)
{
  int n = 4096;
  int iter_max = 1000;
  float *A, *temp;
    
  const float tol = 1.0e-5f;
  //float error_address;
  float error = 1.0f;    

  // get runtime arguments 
  if (argc>1) {  n        = atoi(argv[1]); }
  if (argc>2) {  iter_max = atoi(argv[2]); }

  A    = (float*) malloc( n*n*sizeof(float) );
  temp = (float*) malloc( n*n*sizeof(float) );



  //  set boundary conditions
  laplace_init (A, n);
  laplace_init (temp, n);
  A[(n/128)*n+n/128] = 1.0f; // set singular point
  
  printf("Jacobi relaxation Calculation: %d x %d mesh,"
         " maximum of %d iterations\n", 
         n, n, iter_max );

  // we set the needed number of threads
  omp_set_num_threads(8);

  int iter = 0;
  // in this command,we introduce the parallel region
  // the variables n for the problem size and the maximal interation number iter_max stay
  // the same among the threads, thatswhy they can be declared shared
  // the other variables are thread-specific and every thread should handle his own copy of them
  #pragma omp parallel default(none) firstprivate(iter,temp,A,error) shared(n, iter_max) 
  {
  while (error > tol*tol && iter < iter_max )
  {
    iter++;
    error = laplace_step (A, temp, n);
    float *swap= A; A=temp; temp= swap; // swap pointers A & temp
  }

  // before we output the results, it is necessary to join the threads and only let the master operate
  // to get only one single output and not for every thread each
  #pragma omp master
  {	
  error = sqrtf( error );
  printf("Total Iterations: %5d, ERROR: %0.6f, ", iter, error);
  printf("A[%d][%d]= %0.6f\n", n/128, n/128, A[(n/128)*n+n/128]);
  }
}
  free(A); free(temp);
}



