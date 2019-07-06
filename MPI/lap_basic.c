/* basic MPI version */
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

//laplace step, taking the input and output matrix, the problem dimension, 
// the current number of the process, the total number of processes and
// the entries where the first and last row of a submatrix starts
float laplace_step(float *in, float *out, int n, int me, int nprocs, int ri, int rf)
{
  int i, j;
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
  
 //double for-loop performing the stencil (rf-ri)*n times
  for ( j=ri; j < rf; j+=n ){
	for ( i=1; i < n-1; i++ ){
	  out[j+i]= stencil(in[j+i+1], in[j+i-1], in[(j-n)+i], in[(j+n)+i]);
	  error = max_error( error, out[j+i], in[j+i] );
	}
  }

  return error;
}

// since the initial values have to be calculated globally and not new for every process, we
// have to take into account the number of the process (me) and by this fill the initial values
// following the given boundary conditions
void laplace_init(float *in, int n, int rows_pp, int me)
{
  int i;
  const float pi  = 2.0f * asinf(1.0f);
  memset(in, 0, n*(rows_pp+2)*sizeof(float));
  for (i=0; i<rows_pp; i++) {
    float V = in[(i+1)*n] = sinf(pi*(me*rows_pp+i) / (n-1));
    in[(i+1)*n+( n -1) ] = V*expf(-pi);
  }
}

int main(int argc, char** argv)
{
 
  int n = 4096;
  int iter_max = 1000;
  float *A, *temp;
   
  const float tol = 1.0e-5f;
  float g_error= 1.0f;
  float l_error= 0.0f;    

  // get runtime arguments
  if (argc>1) {  n        = atoi(argv[1]); }
  if (argc>2) {  iter_max = atoi(argv[2]); }

  int me, nprocs;
  
  // we introduce a MPI_Status variable and initialize the MPI communication
  MPI_Status status;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  //Initial Row and Final Rows indices
  //beginning of the first actual row: skipping the k additional rows
  int ri = n; 
  //beginning of the last actual row
  int rf = n * n/nprocs;

  //Rows per process
  int rows_pp = n / nprocs ;

  //reserve space for the actual matrices plus space for k additional rows 
  A    = (float*) malloc( n*(rows_pp + 2)*sizeof(float) );
  temp = (float*) malloc( n*(rows_pp + 2)*sizeof(float) );

  //  set boundary conditions
  laplace_init (A, n, rows_pp, me);
  laplace_init (temp, n, rows_pp, me);
  
  // set singular point
  if(me ==0){
	A[((n/128)+1)*n +n/128] = 1.0f; 
  }
 
  // the information should only be printed by the first process
  if (me == 0){
  printf("Jacobi relaxation Calculation: %d x %d mesh,"
         " maximum of %d iterations\n",
         n, n, iter_max );
  }

  int iter = 0;
  while ( g_error > tol*tol && iter < iter_max )
  {
    iter++;
	
	//for all but the first process:
	if (me > 0){
		// send first row (second in this case, taking into account first row reserved
		// for row to be received from me - 1) to me -1. Tag = 0
		MPI_Send(A+ri,n, MPI_FLOAT,me-1,0, MPI_COMM_WORLD);
		// Receive last row (last but one) from me - 1
		MPI_Recv(A, n, MPI_FLOAT, me-1, 1, MPI_COMM_WORLD, &status);
	}
	// for all but the last process:
	if (me < nprocs -1){
		//send information stored in the last row 
		MPI_Send(A+rf, n, MPI_FLOAT, me+1, 1, MPI_COMM_WORLD);
		//receive information and store them in the additional last row
		MPI_Recv(A+rf+n, n, MPI_FLOAT, me+1, 0, MPI_COMM_WORLD, &status);
	}
	
	l_error= laplace_step (A, temp, n, me, nprocs, ri, rf);
        
	// swap pointers A & temp
    float *swap= A; A=temp; temp= swap; 
    MPI_Allreduce(&l_error, &g_error, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
  }
  //conclude the MPI communication 
  MPI_Finalize();
  g_error = sqrtf( g_error );
  if(me==0){
  printf("Process Nr.:%d\n",me);
  printf("Total Iterations: %5d, ERROR: %0.6f, ", iter, g_error);
  printf("A[%d][%d]= %0.6f\n", (n)/128, (n)/128, A[((n/128)+1)*n+(n)/128]);
  printf("Real position: A[%d][%d]= %0.6f\n", (n)/128+1, (n)/128, A[((n/128)+1)*n+(n)/128]);
  
  /*
  int i,j;
  for(i=0; i<10;i++){
	  for(j=0;j<10;j++){
		  printf("%0.6f ",A[i*n+j]);
		  if(j==9) printf("\n");
	  }
  }/*

}
  free(A); free(temp);

 //MPI_Finalize();
}

