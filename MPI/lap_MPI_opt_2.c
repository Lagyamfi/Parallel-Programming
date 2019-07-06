/* optimized version with block communication */
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

//calculates the next value for an entry by using a mean of the neighours
float stencil ( float v1, float v2, float v3, float v4)
{
  return (v1 + v2 + v3 + v4) * 0.25f;
}

//update of the error
float max_error ( float prev_error, float old, float new )
{
  float t = fabsf( new - old );
  return t>prev_error? t: prev_error;
}

//laplace step, which is only called every k steps
float laplace_step(float *in, float *out, int n, int me, int nprocs, int ri, int rf)
{
  int i, j, cnt;
  float error=0.0f;
  
  // if this is called by the first process, we start one row below, since the first
  // row consists of the constant initial values
  if(me==0){
	  ri+=n;
  }
  // same for the last process: we omit the last row containing constant values
  else if (me==nprocs -1){
	  rf-=n;
  }
  // we do the laplace step like in the basic MPI version, but iterate this step k times
  // k is the depth of block communication
  // in each iteration, we decrease our calculation region by one row at the top and at the bottom
  // so after k steps, we used all the k communicated rows and updated all entries in the inner matrix

	  for ( j=ri; j < rf; j+=n ){
		for ( i=1; i < n-1; i++ ){
		  out[j+i]= stencil(in[j+i+1], in[j+i-1], in[(j-n)+i], in[(j+n)+i]);
		  error = max_error( error, out[j+i], in[j+i] );
		}
	  }
  // the number of computed elements increases by 2*(k-1)*n (by the factor 2*k*n)
  // if(me==0)
  // for(i=0; i<10;i++){
	  // for(j=0;j<10;j++){
		  // printf("%0.6f ",out[i*n+j]);
		  // if(j==9) printf("\n");
	  // }
  // }
  return error;
}

// since the initial values have to be calculated globally and not new for every process, we
// have to take into account the number of the process (me) and by this fill the initial values
// following the given boundary conditions
void laplace_init(float *in, int n, int rows_pp, int me, int k)
{
  int i;
  const float pi  = 2.0f * asinf(1.0f);
  memset(in, 0, n*(rows_pp+2*k)*sizeof(float));
  for (i=0; i<rows_pp; i++) {
    float V = in[(i+k)*n] = sinf(pi*(me*rows_pp+i) / (n-1));
    in[(i+k)*n+(n - 1) ] = V*expf(-pi);
  }
}

int main(int argc, char** argv)
{
 
  int n = 4096;
  int iter_max = 1000;
  float *A, *temp;
  int k = 2; // shared rows
   
  const float tol = 1.0e-5f;
  float g_error= 1.0f;
  float l_error= 0.0f;    

  // get runtime arguments
  if (argc>1) {  n        = atoi(argv[1]); }
  if (argc>2) {  iter_max = atoi(argv[2]); }

  int me, nprocs;
  MPI_Status status;
  MPI_Request request;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  //Initial Row and Final Rows indices
  //beginning of the first actual row: skipping the k additional rows
  int ri = n * k; 
  //beginning of the last actual row
  int rf = n * n/nprocs + (n * (k-1));

  //Rows per process
  int rows_pp = n / nprocs ;

  //reserve space for the actual matrices plus space for k additional rows 
  A    = (float*) malloc( n*(rows_pp + 2*k)*sizeof(float) );
  temp = (float*) malloc( n*(rows_pp + 2*k)*sizeof(float) );

  //set boundary conditions
  laplace_init (A, n, rows_pp, me, k);
  laplace_init (temp, n, rows_pp, me, k);
  if(me==0){
	A[((n/128)+k)*n +n/128] = 1.0f; // set singular point
  }
 
  if (me == 0){
  printf("Jacobi relaxation Calculation: %d x %d mesh,"
         " maximum of %d iterations\n",
         n, n, iter_max );
  }

  int iter = 0;
  while ( g_error > tol*tol && iter < iter_max )
  {
    iter++;
    if (((iter - 1) % k) == 0){
		if (me > 0){
			// send first rows
			MPI_Isend(A+ri,k*n, MPI_FLOAT,me-1,0, MPI_COMM_WORLD, &request);
			// Receive last rows( last but one from me - 1)
			MPI_Irecv(A, k*n, MPI_FLOAT, me-1, 1, MPI_COMM_WORLD, &request);
		}
		if (me < nprocs -1){
			MPI_Isend(A+rf-(k-1)*n, n*k, MPI_FLOAT, me+1, 1, MPI_COMM_WORLD, &request);
			MPI_Irecv(A+rf+n, n*k, MPI_FLOAT, me+1, 0, MPI_COMM_WORLD, &request);
		}
		MPI_Wait(&request, &status );

		//perform the laplace step k times with different start and end rows
		for(int cnt = 0; cnt < k; cnt++){
			l_error= laplace_step (A, temp, n, me, nprocs, ri-((k-1)-cnt)*n, rf+((k-1)-cnt+1)*n);
		}
	    float *swap= A; A=temp; temp= swap; 
		MPI_Allreduce(&l_error, &g_error, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
		
    }     
  }

  MPI_Finalize();
  g_error = sqrtf( g_error );
  if(me==0){
  printf("Process Nr.:%d\n",me);
  printf("Total Iterations: %5d, ERROR: %0.6f, ", iter, g_error);
  printf("A[%d][%d]= %0.6f\n", (n)/128, (n)/128, A[((n/128)+k)*n+(n)/128]);
  printf("Real position: A[%d][%d]= %0.6f\n", (n)/128+k, (n)/128, A[((n/128)+k)*n+(n)/128+1]);
  
  
  int i,j;
  for(i=0; i<10;i++){
	  for(j=0;j<10;j++){
		  printf("%0.6f ",A[i*n+j]);
		  if(j==9) printf("\n");
	  }
  }

}
 free(A); free(temp);

}

