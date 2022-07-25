#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <numa.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>

/*
mpicc -Ofast -ffast-math IFEM_parallel.c -o paralel_INF_Triang -lm -lnuma -fopenmp
mpiexec -n 8 -genv OMP_NUM_THREADS 4 -bind-to socket -map-by socket ./paralel_INF_Triang -env OMP_PROC_BIND true > OUT_paralel_TRIAN.dat &
*/

/*-------------SET-----------------*/

#define OPEN_file_dg "quad_ellipsoid_WGS84_BLH_7_dg.dat"
#define OPEN_file_H "quad_ellipsoid_WGS84_BLH_7.dat"
#define OPEN_write "WGS84_BLH_7_triang_inf"

#define Ne1 512	        	/*number of elements in L direction*/
#define Ne2 128		    	/*number of elements in B direction*/
#define Ne3 57      		/*number of elements in H direction*/

#define Hu 6000000.0    	/*height of finite-element computational domain*/

#define Hsat 200000.0 		/*writing the results above Earth's surface - chosen height */
#define Nu 5     		/*writing the results above Earth's surface - number of nodes H directionfor chosen height*/

#define GaussPoints 2		 /*number of Gauss integration points - finite elements*/
#define GaussPointm 2		 /*number of Gauss integration points - infinite elements*/

double Wi[2]={0.408248290463863,  0.408248290463863};
double Xi[2]={0.166666666666667, 0.666666666666667};
double Wim[2]={1.0000000000000000, 1.0000000000000000};
double Xim[2]={-0.5773502691896257, 0.5773502691896257};

/*Parameters for Solver*/
#define pertol 50 		/*number of itterations for writing residuum*/
#define pertol_write 1000 	/*number of itterations for writing results*/

#define tol 1.0e-5      	/*stopping criterion*/
#define max_it 50000		/*number of itterations*/

/*---------------------------------*/
#define RAD M_PI/180
#define a 6378137.0		/*WGS84 - the semi-major axis */
#define e2 0.006694380004260827 /*WGS84 - the first eccentricity squared*/ 

struct Nodes
{
	double x,y,z;
};

int	myid,nprocs;
long global_vmrss, global_vmsize;
int64_t    prev, next, istart, iglobal, iter, n, N1, n1, n1g, ne1, N2, n2, ne2, n3, ne3, dwe, dsn, n1p, n2p, n3p;
double pom,res,start,end,tmp,men,sum,height;
double *x,*b,*MC,*B,*L,*H;
struct Nodes *node;
char *name;
FILE *fr, *fw;

MPI_Status status;
MPI_Request req1,req2,req3,req4;

void XYZnodes()
{
 	int64_t i, j, k, kk, ind, from,to;
 	double N, M, tmp; 
	
	B = (double*) malloc(N1 * n2p* sizeof(double));
	L = (double*) malloc(N1 * n2p* sizeof(double));
	H = (double*) malloc(N1 * n2p* sizeof(double));
 	
 	if (myid == 0) 
	{	
		printf("Reading H BC ");
		
		fr = fopen(OPEN_file_H,"r");
	
		for (j = 1; j <= n2; j++)
			for (i = 0; i < N1; i++)
				{
					ind = (i*n2p+j);
					fscanf(fr, "%lf", &B[ind]);
					fscanf(fr, "%lf", &L[ind]);
					fscanf(fr, "%lf", &H[ind]);
				}
		fclose(fr);
	
		for (i = 0; i < N1; i++)
				{
					B[(N1-i-1)*n2p+0]=B[i*n2p+2];
					L[(N1-i-1)*n2p+0]=L[i*n2p+2];
					H[(N1-i-1)*n2p+0]=H[i*n2p+2];					
				}
		
		for (i = 0; i < N1; i++)
		{

			if ( i<=(int)((double)Ne1/2.0))
				{
					B[((int)((double)Ne1/2.0)-i)*n2p+n2+1]=B[i*n2p+n2-1];
					L[((int)((double)Ne1/2.0)-i)*n2p+n2+1]=L[i*n2p+n2-1];
					H[((int)((double)Ne1/2.0)-i)*n2p+n2+1]=H[i*n2p+n2-1];	
				}
			else 
				{
					B[(N1-i-1+(int)((double)Ne1/2.0))*n2p+n2+1]=B[i*n2p+n2-1];
					L[(N1-i-1+(int)((double)Ne1/2.0))*n2p+n2+1]=L[i*n2p+n2-1];
					H[(N1-i-1+(int)((double)Ne1/2.0))*n2p+n2+1]=H[i*n2p+n2-1];
				}
		}
		B[(N1-1)*n2p+n2+1]=B[0*n2p+n2+1];
		L[(N1-1)*n2p+n2+1]=L[0*n2p+n2+1];
		H[(N1-1)*n2p+n2+1]=H[0*n2p+n2+1];
		
		printf(" OK \n");	
	}
	
	MPI_Bcast(B,N1 * n2p, MPI_DOUBLE,0, MPI_COMM_WORLD);
	MPI_Bcast(L,N1 * n2p, MPI_DOUBLE,0, MPI_COMM_WORLD);
	MPI_Bcast(H,N1 * n2p, MPI_DOUBLE,0, MPI_COMM_WORLD);

	if (myid == 0) 
		printf("Compute nodes coordinates \n ");

	//#pragma omp parallel for private(j,k,kk,ind,iglobal,N,deltaH,height) 		
	for (i=0;i<=n1+1;i++)
		for (j=0;j<=n2+1;j++)
			for (k=0;k<n3;k++)
			{
				ind = ((i*n2p+j)*n3p+k);
				if (myid ==0 && i==0)
					iglobal= n2p*(N1-2)+j; 
				else if (myid == nprocs - 1 && i==n1+1)
					iglobal= n2p*(1)+j; 
				else
					iglobal= n2p*(i-1)+j+(myid*((n1g-1)*n2p));			

			N =(a/sqrt(1-e2*sin(B[iglobal]*RAD)*sin(B[iglobal]*RAD)));	

			if(k<=Nu)
			{
				height=H[iglobal]+k*((Hsat-H[iglobal])/(Nu));
			}
			
			else if (k>Nu)
			{

				height=Hsat+(k-Nu)*((Hu-Hsat)/(n3-3-Nu));
			}			
			else if (k>=n3-2)
				height=2*Hu+N;
			
			if ((myid == 0) && (i==1) && (j==1)&& (k<(Nu+2)))
				printf("%d %lf\n",k,height);
			if ((myid == 0) && (i==1) && (j==1) && (k>=n3-5) && (k<=n3-2))
				printf("%d %lf\n",k,height);	
					
			node[ind].x=(N+height)*cos(B[iglobal]*RAD)*cos(L[iglobal]*RAD);
			node[ind].y=(N+height)*cos(B[iglobal]*RAD)*sin(L[iglobal]*RAD);
			node[ind].z=((1-e2)*N+height)*sin(B[iglobal]*RAD);	
			}	
			
	if (myid == 0) 
		printf(" OK \n");	

 }
 
void trojA(int64_t i,int64_t j, int64_t k, int64_t ind) 
{
	
	int64_t index;
	int ii,jj,kk;
	
	double e,n,m,determ,invdet;
	double X1,X2,X3,X4,X5,X6,X7,X8;
	double Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8;
	double Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8;
	double g11,g12,g13,g21,g22,g23,g31,g32,g33;
	
	double Jacobi[3][3], A[3][12], K[12][12];
					
	X1=node[ind].x;    X2=node[ind+dwe].x; 	 X3=node[ind+dsn].x;
    	X4=node[ind+1].x;  X5=node[ind+dwe+1].x; X6=node[ind+dsn+1].x;
   
    	Y1=node[ind].y;    Y2=node[ind+dwe].y; 	Y3=node[ind+dsn].y;
    	Y4=node[ind+1].y;    Y5=node[ind+dwe+1].y; 	Y6=node[ind+dsn+1].y;
    
	Z1=node[ind].z;    Z2=node[ind+dwe].z; 	Z3=node[ind+dsn].z;
    	Z4=node[ind+1].z;    Z5=node[ind+dwe+1].z; 	Z6=node[ind+dsn+1].z;
								
	for (ii=0;ii<GaussPoints;ii++)
		for (jj=0;jj<GaussPoints;jj++)
			for (kk=0;kk<GaussPointm;kk++)
			{
				e=Xi[ii];
				n=Xi[jj];
				m=Xim[kk];
								

    Jacobi[0][0]=-1/2.0*(1-m)*X1+1/2.0*(1-m)*X2+0*X3+(-1/2.0)*(1+m)*X4+1/2.0*(1+m)*X5+0*X6;
    Jacobi[0][1]=-1/2.0*(1-m)*Y1+1/2.0*(1-m)*Y2+0*Y3+(-1/2.0)*(1+m)*Y4+1/2.0*(1+m)*Y5+0*Y6;
    Jacobi[0][2]=-1/2.0*(1-m)*Z1+1/2.0*(1-m)*Z2+0*Z3+(-1/2.0)*(1+m)*Z4+1/2.0*(1+m)*Z5+0*Z6;
    Jacobi[1][0]= -1/2.0*(1-m)*X1+0*X2+1/2.0*(1-m)*X3+(-1/2.0)*(1+m)*X4+0*X5+1/2.0*(1+m)*X6;
    Jacobi[1][1]= -1/2.0*(1-m)*Y1+0*Y2+1/2.0*(1-m)*Y3+(-1/2.0)*(1+m)*Y4+0*Y5+1/2.0*(1+m)*Y6;
    Jacobi[1][2]= -1/2.0*(1-m)*Z1+0*Z2+1/2.0*(1-m)*Z3+(-1/2.0)*(1+m)*Z4+0*Z5+1/2.0*(1+m)*Z6;
    Jacobi[2][0]= 1/2.0*(-1+e+n)*X1-e*X2/2.0-n*X3/2.0+1/2.0*(1-e-n)*X4+e*X5/2.0+n*X6/2.0;
    Jacobi[2][1]= 1/2.0*(-1+e+n)*Y1-e*Y2/2.0-n*Y3/2.0+1/2.0*(1-e-n)*Y4+e*Y5/2.0+n*Y6/2.0;
    Jacobi[2][2]= 1/2.0*(-1+e+n)*Z1-e*Z2/2.0-n*Z3/2.0+1/2.0*(1-e-n)*Z4+e*Z5/2.0+n*Z6/2.0;   

	determ = Jacobi[0][0]*((Jacobi[1][1]*Jacobi[2][2]) - (Jacobi[2][1]*Jacobi[1][2])) -Jacobi[0][1]*(Jacobi[1][0]*Jacobi[2][2] - Jacobi[2][0]*Jacobi[1][2]) + Jacobi[0][2]*(Jacobi[1][0]*Jacobi[2][1] - Jacobi[2][0]*Jacobi[1][1]);
		
 	invdet = 1/determ;
	
	if ((ii==GaussPoints-1)&&(jj==GaussPoints-1))
    {	e=0.0;    n=0.0;    m=0.0; invdet=0.0; determ=0.0;}
 	
	g11 = (Jacobi[1][1] * Jacobi[2][2] - Jacobi[2][1] * Jacobi[1][2]) * invdet;
	g12 = (Jacobi[0][2] * Jacobi[2][1] - Jacobi[0][1] * Jacobi[2][2]) * invdet;
	g13 = (Jacobi[0][1] * Jacobi[1][2] - Jacobi[0][2] * Jacobi[1][1]) * invdet;
	g21 = (Jacobi[1][2] * Jacobi[2][0] - Jacobi[1][0] * Jacobi[2][2]) * invdet;
	g22 = (Jacobi[0][0] * Jacobi[2][2] - Jacobi[0][2] * Jacobi[2][0]) * invdet;
	g23 = (Jacobi[1][0] * Jacobi[0][2] - Jacobi[0][0] * Jacobi[1][2]) * invdet;
	g31 = (Jacobi[1][0] * Jacobi[2][1] - Jacobi[2][0] * Jacobi[1][1]) * invdet;
	g32 = (Jacobi[2][0] * Jacobi[0][1] - Jacobi[0][0] * Jacobi[2][1]) * invdet;
	g33 = (Jacobi[0][0] * Jacobi[1][1] - Jacobi[1][0] * Jacobi[0][1]) * invdet;
		
	A[0][0] = -1/2.0*(1-m)*g11-1/2.0*(1-m)*g12+1/2.0*(-1+e+n)*g13;  
	A[0][1] = 1/2.0*(1-m)*g11+0.0*g12-e*g13/2.0;  
	A[0][2] = 0.0*g11+1/2.0*(1-m)*g12-n*g13/2.0;  
	A[0][3] = (-1/2.0)*(1+m)*g11+(-1/2.0)*(1+m)*g12+1/2.0*(1-e-n)*g13;
	A[0][4] = 1/2.0*(1+m)*g11+0.0*g12+e*g13/2.0; 
	A[0][5] = 0.0*g11+1/2.0*(1+m)*g12+n*g13/2.0;  
		
	A[1][0] = -1/2.0*(1-m)*g21-1/2.0*(1-m)*g22+1/2.0*(-1+e+n)*g23;
	A[1][1] = 1/2.0*(1-m)*g21+0.0*g22-e*g23/2.0;
	A[1][2] = 0.0*g21+1/2.0*(1-m)*g22-n*g23/2.0;
	A[1][3] = (-1/2.0)*(1+m)*g21+(-1/2.0)*(1+m)*g22+1/2.0*(1-e-n)*g23;
	A[1][4] = 1/2.0*(1+m)*g21+0.0*g22+e*g23/2.0;
	A[1][5] = 0.0*g21+1/2.0*(1+m)*g22+n*g23/2.0;
	
	A[2][0] = -1/2.0*(1-m)*g31-1/2.0*(1-m)*g32+1/2.0*(-1+e+n)*g33;
	A[2][1] = 1/2.0*(1-m)*g31+0.0*g32-e*g33/2.0;
	A[2][2] = 0.0*g31+1/2.0*(1-m)*g32-n*g33/2.0;
	A[2][3] = (-1/2.0)*(1+m)*g31+(-1/2.0)*(1+m)*g32+1/2.0*(1-e-n)*g33;
	A[2][4] = 1/2.0*(1+m)*g31+0.0*g32+e*g33/2.0;
	A[2][5] = 0.0*g31+1/2.0*(1+m)*g32+n*g33/2.0;

index = ((i*n2p + j)*n3p + k)*36+13;	
	
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][0]*A[0][0]+A[1][0]*A[1][0]+A[2][0]*A[2][0]);
	MC[index+9]=MC[index+9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][1]+A[1][0]*A[1][1]+A[2][0]*A[2][1]);
	MC[index+3]=MC[index+3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][2]+A[1][0]*A[1][2]+A[2][0]*A[2][2]);
	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][3]+A[1][0]*A[1][3]+A[2][0]*A[2][3]);
	MC[index+10]=MC[index+10]+Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][4]+A[1][0]*A[1][4]+A[2][0]*A[2][4]);
	MC[index+4]=MC[index+4] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][5]+A[1][0]*A[1][5]+A[2][0]*A[2][5]);

index = (((i+1)*n2p + j)*n3p + k)*36+13;

	MC[index-9]=MC[index-9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][0]+A[1][1]*A[1][0]+A[2][1]*A[2][0]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][1]*A[0][1]+A[1][1]*A[1][1]+A[2][1]*A[2][1]);
	MC[index-6]=MC[index-6] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][2]+A[1][1]*A[1][2]+A[2][1]*A[2][2]);
	MC[index-8]=MC[index-8] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][3]+A[1][1]*A[1][3]+A[2][1]*A[2][3]);
	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][4]+A[1][1]*A[1][4]+A[2][1]*A[2][4]);
	MC[index-5]=MC[index-5] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][5]+A[1][1]*A[1][5]+A[2][1]*A[2][5]);

index = ((i*n2p + j+1)*n3p + k)*36+13;

	MC[index-3]=MC[index-3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][0]+A[1][2]*A[1][0]+A[2][2]*A[2][0]);
	MC[index+6]=MC[index+6] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][1]+A[1][2]*A[1][1]+A[2][2]*A[2][1]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][2]*A[0][2]+A[1][2]*A[1][2]+A[2][2]*A[2][2]);
	MC[index-2]=MC[index-2] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][3]+A[1][2]*A[1][3]+A[2][2]*A[2][3]);
	MC[index+7]=MC[index+7] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][4]+A[1][2]*A[1][4]+A[2][2]*A[2][4]);
	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][5]+A[1][2]*A[1][5]+A[2][2]*A[2][5]);

index = ((i*n2p + j)*n3p + k+1)*36+13;
	
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][0]+A[1][3]*A[1][0]+A[2][3]*A[2][0]);
	MC[index+8]=MC[index+8] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][1]+A[1][3]*A[1][1]+A[2][3]*A[2][1]);
	MC[index+2]=MC[index+2] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][2]+A[1][3]*A[1][2]+A[2][3]*A[2][2]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][3]*A[0][3]+A[1][3]*A[1][3]+A[2][3]*A[2][3]);
	MC[index+9]=MC[index+9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][4]+A[1][3]*A[1][4]+A[2][3]*A[2][4]);
	MC[index+3]=MC[index+3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][5]+A[1][3]*A[1][5]+A[2][3]*A[2][5]);


index = (((i+1)*n2p + j)*n3p + k+1)*36+13;
	
	MC[index-10]=MC[index-10]+Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][0]+A[1][4]*A[1][0]+A[2][4]*A[2][0]);
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][1]+A[1][4]*A[1][1]+A[2][4]*A[2][1]);
	MC[index-7]=MC[index-7] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][2]+A[1][4]*A[1][2]+A[2][4]*A[2][2]);
	MC[index-9]=MC[index-9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][3]+A[1][4]*A[1][3]+A[2][4]*A[2][3]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][4]*A[0][4]+A[1][4]*A[1][4]+A[2][4]*A[2][4]);
	MC[index-6]=MC[index-6] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][5]+A[1][4]*A[1][5]+A[2][4]*A[2][5]);

index = ((i*n2p + j+1)*n3p + k+1)*36+13;
	
	MC[index-4]=MC[index-4] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][0]+A[1][5]*A[1][0]+A[2][5]*A[2][0]);
	MC[index+5]=MC[index+5] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][1]+A[1][5]*A[1][1]+A[2][5]*A[2][1]);
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][2]+A[1][5]*A[1][2]+A[2][5]*A[2][2]);
	MC[index-3]=MC[index-3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][3]+A[1][5]*A[1][3]+A[2][5]*A[2][3]);
	MC[index+6]=MC[index+6] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][4]+A[1][5]*A[1][4]+A[2][5]*A[2][4]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][5]*A[0][5]+A[1][5]*A[1][5]+A[2][5]*A[2][5]);
				}					
						
	X1=node[ind+dwe+dsn].x;    X2=node[ind+dsn].x;    X3=node[ind+dwe].x; 	 
    X4=node[ind+dwe+dsn+1].x;  X5=node[ind+dsn+1].x;  X6=node[ind+dwe+1].x; 
   
	Y1=node[ind+dwe+dsn].y;    Y2=node[ind+dsn].y;    Y3=node[ind+dwe].y; 	 
    Y4=node[ind+dwe+dsn+1].y;  Y5=node[ind+dsn+1].y;  Y6=node[ind+dwe+1].y; 
    
	Z1=node[ind+dwe+dsn].z;    Z2=node[ind+dsn].z;    Z3=node[ind+dwe].z; 	 
    Z4=node[ind+dwe+dsn+1].z;  Z5=node[ind+dsn+1].z;  Z6=node[ind+dwe+1].z;
	
	for (ii=0;ii<GaussPoints;ii++)
		for (jj=0;jj<GaussPoints;jj++)
			for (kk=0;kk<GaussPointm;kk++)
			{
				e=Xi[ii];
				n=Xi[jj];
				m=Xim[kk];
 
    Jacobi[0][0]=-1/2.0*(1-m)*X1+1/2.0*(1-m)*X2+0*X3+(-1/2.0)*(1+m)*X4+1/2.0*(1+m)*X5+0*X6;
    Jacobi[0][1]=-1/2.0*(1-m)*Y1+1/2.0*(1-m)*Y2+0*Y3+(-1/2.0)*(1+m)*Y4+1/2.0*(1+m)*Y5+0*Y6;
    Jacobi[0][2]=-1/2.0*(1-m)*Z1+1/2.0*(1-m)*Z2+0*Z3+(-1/2.0)*(1+m)*Z4+1/2.0*(1+m)*Z5+0*Z6;
    Jacobi[1][0]= -1/2.0*(1-m)*X1+0*X2+1/2.0*(1-m)*X3+(-1/2.0)*(1+m)*X4+0*X5+1/2.0*(1+m)*X6;
    Jacobi[1][1]= -1/2.0*(1-m)*Y1+0*Y2+1/2.0*(1-m)*Y3+(-1/2.0)*(1+m)*Y4+0*Y5+1/2.0*(1+m)*Y6;
    Jacobi[1][2]= -1/2.0*(1-m)*Z1+0*Z2+1/2.0*(1-m)*Z3+(-1/2.0)*(1+m)*Z4+0*Z5+1/2.0*(1+m)*Z6;
    Jacobi[2][0]= 1/2.0*(-1+e+n)*X1-e*X2/2.0-n*X3/2.0+1/2.0*(1-e-n)*X4+e*X5/2.0+n*X6/2.0;
    Jacobi[2][1]= 1/2.0*(-1+e+n)*Y1-e*Y2/2.0-n*Y3/2.0+1/2.0*(1-e-n)*Y4+e*Y5/2.0+n*Y6/2.0;
    Jacobi[2][2]= 1/2.0*(-1+e+n)*Z1-e*Z2/2.0-n*Z3/2.0+1/2.0*(1-e-n)*Z4+e*Z5/2.0+n*Z6/2.0;   

	determ = Jacobi[0][0]*((Jacobi[1][1]*Jacobi[2][2]) - (Jacobi[2][1]*Jacobi[1][2])) -Jacobi[0][1]*(Jacobi[1][0]*Jacobi[2][2] - Jacobi[2][0]*Jacobi[1][2]) + Jacobi[0][2]*(Jacobi[1][0]*Jacobi[2][1] - Jacobi[2][0]*Jacobi[1][1]);
		
 	invdet = 1/determ;
	
	if ((ii==GaussPoints-1)&&(jj==GaussPoints-1))
    {	e=0.0;    n=0.0;    m=0.0; invdet=0.0; determ=0.0;}
	 	
	g11 = (Jacobi[1][1] * Jacobi[2][2] - Jacobi[2][1] * Jacobi[1][2]) * invdet;
	g12 = (Jacobi[0][2] * Jacobi[2][1] - Jacobi[0][1] * Jacobi[2][2]) * invdet;
	g13 = (Jacobi[0][1] * Jacobi[1][2] - Jacobi[0][2] * Jacobi[1][1]) * invdet;
	g21 = (Jacobi[1][2] * Jacobi[2][0] - Jacobi[1][0] * Jacobi[2][2]) * invdet;
	g22 = (Jacobi[0][0] * Jacobi[2][2] - Jacobi[0][2] * Jacobi[2][0]) * invdet;
	g23 = (Jacobi[1][0] * Jacobi[0][2] - Jacobi[0][0] * Jacobi[1][2]) * invdet;
	g31 = (Jacobi[1][0] * Jacobi[2][1] - Jacobi[2][0] * Jacobi[1][1]) * invdet;
	g32 = (Jacobi[2][0] * Jacobi[0][1] - Jacobi[0][0] * Jacobi[2][1]) * invdet;
	g33 = (Jacobi[0][0] * Jacobi[1][1] - Jacobi[1][0] * Jacobi[0][1]) * invdet;
		
	A[0][0] = -1/2.0*(1-m)*g11-1/2.0*(1-m)*g12+1/2.0*(-1+e+n)*g13;  
	A[0][1] = 1/2.0*(1-m)*g11+0.0*g12-e*g13/2.0;  
	A[0][2] = 0.0*g11+1/2.0*(1-m)*g12-n*g13/2.0;  
	A[0][3] = (-1/2.0)*(1+m)*g11+(-1/2.0)*(1+m)*g12+1/2.0*(1-e-n)*g13;
	A[0][4] = 1/2.0*(1+m)*g11+0.0*g12+e*g13/2.0; 
	A[0][5] = 0.0*g11+1/2.0*(1+m)*g12+n*g13/2.0; 
	
	A[1][0] = -1/2.0*(1-m)*g21-1/2.0*(1-m)*g22+1/2.0*(-1+e+n)*g23;
	A[1][1] = 1/2.0*(1-m)*g21+0.0*g22-e*g23/2.0;
	A[1][2] = 0.0*g21+1/2.0*(1-m)*g22-n*g23/2.0;
	A[1][3] = (-1/2.0)*(1+m)*g21+(-1/2.0)*(1+m)*g22+1/2.0*(1-e-n)*g23;
	A[1][4] = 1/2.0*(1+m)*g21+0.0*g22+e*g23/2.0;
	A[1][5] = 0.0*g21+1/2.0*(1+m)*g22+n*g23/2.0;
	
	A[2][0] = -1/2.0*(1-m)*g31-1/2.0*(1-m)*g32+1/2.0*(-1+e+n)*g33;
	A[2][1] = 1/2.0*(1-m)*g31+0.0*g32-e*g33/2.0;
	A[2][2] = 0.0*g31+1/2.0*(1-m)*g32-n*g33/2.0;
	A[2][3] = (-1/2.0)*(1+m)*g31+(-1/2.0)*(1+m)*g32+1/2.0*(1-e-n)*g33;
	A[2][4] = 1/2.0*(1+m)*g31+0.0*g32+e*g33/2.0;
	A[2][5] = 0.0*g31+1/2.0*(1+m)*g32+n*g33/2.0;
	
index = (((i+1)*n2p + j+1)*n3p + k)*36+13;
	
	MC[index]   =   MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][0]+A[1][0]*A[1][0]+A[2][0]*A[2][0]);
	MC[index-9] = MC[index-9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][1]+A[1][0]*A[1][1]+A[2][0]*A[2][1]);
	MC[index-3]  =MC[index-3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][2]+A[1][0]*A[1][2]+A[2][0]*A[2][2]);
	MC[index+1] = MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][3]+A[1][0]*A[1][3]+A[2][0]*A[2][3]);
	MC[index-8]  =MC[index-8] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][4]+A[1][0]*A[1][4]+A[2][0]*A[2][4]);
	MC[index-2] = MC[index-2] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][5]+A[1][0]*A[1][5]+A[2][0]*A[2][5]);

index = ((i*n2p + j+1)*n3p + k)*36+13;

	MC[index+9]=MC[index+9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][0]+A[1][1]*A[1][0]+A[2][1]*A[2][0]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][1]*A[0][1]+A[1][1]*A[1][1]+A[2][1]*A[2][1]);
	MC[index+6]=MC[index+6] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][2]+A[1][1]*A[1][2]+A[2][1]*A[2][2]);
	MC[index+10]=MC[index+10]+Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][3]+A[1][1]*A[1][3]+A[2][1]*A[2][3]);
	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][4]+A[1][1]*A[1][4]+A[2][1]*A[2][4]);
	MC[index+7]=MC[index+7] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][5]+A[1][1]*A[1][5]+A[2][1]*A[2][5]);

index = (((i+1)*n2p + j)*n3p + k)*36+13;

	MC[index+3]=MC[index+3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][0]+A[1][2]*A[1][0]+A[2][2]*A[2][0]);
	MC[index-6]=MC[index-6] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][1]+A[1][2]*A[1][1]+A[2][2]*A[2][1]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][2]*A[0][2]+A[1][2]*A[1][2]+A[2][2]*A[2][2]);
	MC[index+4]=MC[index+4] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][3]+A[1][2]*A[1][3]+A[2][2]*A[2][3]);
	MC[index-5]=MC[index-5] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][4]+A[1][2]*A[1][4]+A[2][2]*A[2][4]);
	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][5]+A[1][2]*A[1][5]+A[2][2]*A[2][5]);

index = (((i+1)*n2p + j+1)*n3p + k+1)*36+13;
	
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][0]+A[1][3]*A[1][0]+A[2][3]*A[2][0]);
	MC[index-10]=MC[index-10]+Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][1]+A[1][3]*A[1][1]+A[2][3]*A[2][1]);
	MC[index-4]=MC[index-4] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][2]+A[1][3]*A[1][2]+A[2][3]*A[2][2]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][3]*A[0][3]+A[1][3]*A[1][3]+A[2][3]*A[2][3]);
	MC[index-9]=MC[index-9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][4]+A[1][3]*A[1][4]+A[2][3]*A[2][4]);
	MC[index-3]=MC[index-3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][5]+A[1][3]*A[1][5]+A[2][3]*A[2][5]);

index = ((i*n2p + j+1)*n3p + k+1)*36+13;
	
	MC[index+8]=MC[index+8] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][0]+A[1][4]*A[1][0]+A[2][4]*A[2][0]);
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][1]+A[1][4]*A[1][1]+A[2][4]*A[2][1]);
	MC[index+5]=MC[index+5] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][2]+A[1][4]*A[1][2]+A[2][4]*A[2][2]);
	MC[index+9]=MC[index+9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][3]+A[1][4]*A[1][3]+A[2][4]*A[2][3]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][4]*A[0][4]+A[1][4]*A[1][4]+A[2][4]*A[2][4]);
	MC[index+6]=MC[index+6] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][5]+A[1][4]*A[1][5]+A[2][4]*A[2][5]);
		
	
index = (((i+1)*n2p + j)*n3p + k+1)*36+13;
	
	MC[index+2]=MC[index+2] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][0]+A[1][5]*A[1][0]+A[2][5]*A[2][0]);
	MC[index-7]=MC[index-7] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][1]+A[1][5]*A[1][1]+A[2][5]*A[2][1]);
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][2]+A[1][5]*A[1][2]+A[2][5]*A[2][2]);
	MC[index+3]=MC[index+3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][3]+A[1][5]*A[1][3]+A[2][5]*A[2][3]);
	MC[index-6]=MC[index-6] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][4]+A[1][5]*A[1][4]+A[2][5]*A[2][4]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][5]*A[0][5]+A[1][5]*A[1][5]+A[2][5]*A[2][5]);
	
				}				
				
}
 
void trojInfA(int64_t i,int64_t j, int64_t k, int64_t ind) 
{
	int64_t index;
	int ii,jj,kk;
	
	double e,n,m,determ,invdet;
	double X1,X2,X3,X4,X5,X6,X7,X8;
	double Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8;
	double Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8;
	double g11,g12,g13,g21,g22,g23,g31,g32,g33;
	
	double Jacobi[3][3], A[3][12], K[12][12];
									
	X1=node[ind].x;    X2=node[ind+dwe].x; 	 X3=node[ind+dsn].x;
    X4=node[ind+1].x;  X5=node[ind+dwe+1].x; X6=node[ind+dsn+1].x;
   
    Y1=node[ind].y;    Y2=node[ind+dwe].y; 	Y3=node[ind+dsn].y;
    Y4=node[ind+1].y;    Y5=node[ind+dwe+1].y; 	Y6=node[ind+dsn+1].y;
    
	Z1=node[ind].z;    Z2=node[ind+dwe].z; 	Z3=node[ind+dsn].z;
    Z4=node[ind+1].z;    Z5=node[ind+dwe+1].z; 	Z6=node[ind+dsn+1].z;
	
	for (ii=0;ii<GaussPoints;ii++)
		for (jj=0;jj<GaussPoints;jj++)
			for (kk=0;kk<GaussPointm;kk++)
			{
				e=Xi[ii];
				n=Xi[jj];
				m=Xim[kk];

	Jacobi[0][0]=-((2.0*m)/(m - 1.0))*X1+ ((2.0*m)/(m - 1.0))*X2+0*X3+((m + 1)/((m - 1.0)))*X4 + (-(m + 1)/((m - 1.0)))*X5+0*X6;
	Jacobi[0][1]=-((2.0*m)/(m - 1.0))*Y1+ ((2.0*m)/(m - 1.0))*Y2+ 0*Y3+ ((m + 1)/((m - 1.0)))*Y4 + (-(m + 1)/((m - 1.0)))*Y5+0*Y6;
	Jacobi[0][2]=-((2.0*m)/(m - 1.0))*Z1+ ((2.0*m)/(m - 1.0))*Z2+ 0*Z3+ ((m + 1)/((m - 1.0)))*Z4 + (-(m + 1)/((m - 1.0)))*Z5+0*Z6;
	Jacobi[1][0]=-((2.0*m)/(m - 1.0))*X1+0*X2+ ((2.0*m)/(m - 1.0))*X3+ ((m + 1)/((m - 1.0)))*X4+0*X5+( -(m + 1)/((m - 1.0)))*X6;
	Jacobi[1][1]=-((2.0*m)/(m - 1.0))*Y1+0*Y2+ ((2.0*m)/(m - 1.0))*Y3+ ((m + 1)/((m - 1.0)))*Y4+0*Y5+( -(m + 1)/((m - 1.0)))*Y6;
	Jacobi[1][2]=-((2.0*m)/(m - 1.0))*Z1+0*Z2+ ((2.0*m)/(m - 1.0))*Z3+ ((m + 1)/((m - 1.0)))*Z4+0*Z5+( -(m + 1)/((m - 1.0)))*Z6;
	Jacobi[2][0]=((2.0*m*(e + n - 1.0))/((m - 1.0)*(m - 1.0)) - (2*(e + n - 1.0))/(m - 1.0))*X1+((2.0*e)/(m - 1.0) - (2.0*e*m)/((m - 1.0)*(m - 1.0)))*X2+((2.0*n)/(m - 1.0) - (2.0*m*n)/((m - 1.0)*(m - 1.0)))*X3+( (e + n - 1.0)/(m - 1.0) - ((m + 1)*(e + n - 1.0))/((m - 1.0)*(m - 1.0)))*X4+((e*(m + 1))/((m - 1.0)*(m - 1.0)) - e/(m - 1.0))*X5+((n*(m + 1))/((m - 1.0)*(m - 1.0)) - n/(m - 1.0))*X6;
	Jacobi[2][1]=((2.0*m*(e + n - 1.0))/((m - 1.0)*(m - 1.0)) - (2*(e + n - 1.0))/(m - 1.0))*Y1+((2.0*e)/(m - 1.0) - (2.0*e*m)/((m - 1.0)*(m - 1.0)))*Y2+((2.0*n)/(m - 1.0) - (2.0*m*n)/((m - 1.0)*(m - 1.0)))*Y3+( (e + n - 1.0)/(m - 1.0) - ((m + 1)*(e + n - 1.0))/((m - 1.0)*(m - 1.0)))*Y4+((e*(m + 1))/((m - 1.0)*(m - 1.0)) - e/(m - 1.0))*Y5+((n*(m + 1))/((m - 1.0)*(m - 1.0)) - n/(m - 1.0))*Y6;
	Jacobi[2][2]=((2.0*m*(e + n - 1.0))/((m - 1.0)*(m - 1.0)) - (2*(e + n - 1.0))/(m - 1.0))*Z1+((2.0*e)/(m - 1.0) - (2.0*e*m)/((m - 1.0)*(m - 1.0)))*Z2+((2.0*n)/(m - 1.0) - (2.0*m*n)/((m - 1.0)*(m - 1.0)))*Z3+( (e + n - 1.0)/(m - 1.0) - ((m + 1)*(e + n - 1.0))/((m - 1.0)*(m - 1.0)))*Z4+((e*(m + 1))/((m - 1.0)*(m - 1.0)) - e/(m - 1.0))*Z5+((n*(m + 1))/((m - 1.0)*(m - 1.0)) - n/(m - 1.0))*Z6;  

	determ = Jacobi[0][0]*((Jacobi[1][1]*Jacobi[2][2]) - (Jacobi[2][1]*Jacobi[1][2])) -Jacobi[0][1]*(Jacobi[1][0]*Jacobi[2][2] - Jacobi[2][0]*Jacobi[1][2]) + Jacobi[0][2]*(Jacobi[1][0]*Jacobi[2][1] - Jacobi[2][0]*Jacobi[1][1]);
	
 	invdet = 1/determ;
	 	
	if ((ii==GaussPoints-1)&&(jj==GaussPoints-1))
    {	e=0.0;    n=0.0;    m=0.0; invdet=0.0; determ=0.0;}

	g11 = (Jacobi[1][1] * Jacobi[2][2] - Jacobi[2][1] * Jacobi[1][2]) * invdet;
	g12 = (Jacobi[0][2] * Jacobi[2][1] - Jacobi[0][1] * Jacobi[2][2]) * invdet;
	g13 = (Jacobi[0][1] * Jacobi[1][2] - Jacobi[0][2] * Jacobi[1][1]) * invdet;
	g21 = (Jacobi[1][2] * Jacobi[2][0] - Jacobi[1][0] * Jacobi[2][2]) * invdet;
	g22 = (Jacobi[0][0] * Jacobi[2][2] - Jacobi[0][2] * Jacobi[2][0]) * invdet;
	g23 = (Jacobi[1][0] * Jacobi[0][2] - Jacobi[0][0] * Jacobi[1][2]) * invdet;
	g31 = (Jacobi[1][0] * Jacobi[2][1] - Jacobi[2][0] * Jacobi[1][1]) * invdet;
	g32 = (Jacobi[2][0] * Jacobi[0][1] - Jacobi[0][0] * Jacobi[2][1]) * invdet;
	g33 = (Jacobi[0][0] * Jacobi[1][1] - Jacobi[1][0] * Jacobi[0][1]) * invdet;   	
		
	A[0][0] = -((m*(m - 1))/2.0)*g11-((m*(m - 1))/2.0)*g12-(((m - 1)*(e + n - 1))/2.0 + (m*(e + n - 1))/2.0)*g13;  
	A[0][1] = -(-(m*(m - 1))/2.0)*g11-0.0*g12-(- (e*m)/2.0 - (e*(m - 1))/2.0)*g13;  
	A[0][2] = -0.0*g11-(-(m*(m - 1))/2.0)*g12-(- (m*n)/2.0 - (n*(m - 1))/2.0)*g13;  
	A[0][3] = (m*m-1)*g11+(m*m-1)*g12+(2*m*(e+n-1))*g13;
	A[0][4] = (1-m*m)*g11+0.0*g12+(-2*e*m)*g13; 
	A[0][5] = 0.0*g11+(1-m*m)*g12+(-2*m*n)*g13;  
	A[0][6] = (-(m*(m+1))/2.0)*g11+(-(m*(m+1))/2.0)*g12+(-((m+1)*(e+n-1))/2.0-(m*(e+n-1))/2.0)*g13; 
	A[0][7] = ((m*(m+1))/2.0)*g11+0.0*g12+((e*m)/2.0+(e*(m+1))/2.0)*g13;
	A[0][8] = 0.0*g11+((m*(m+1))/2.0)*g12+((m*n)/2.0+(n*(m+1))/2.0)*g13; 

	A[1][0] = -((m*(m - 1))/2.0)*g21-((m*(m - 1))/2.0)*g22-(((m - 1)*(e + n - 1))/2.0 + (m*(e + n - 1))/2.0)*g23;  
	A[1][1] = -(-(m*(m - 1))/2.0)*g21-0.0*g22-(- (e*m)/2.0 - (e*(m - 1))/2.0)*g23;  
	A[1][2] = -0.0*g21-(-(m*(m - 1))/2.0)*g22-(- (m*n)/2.0 - (n*(m - 1))/2.0)*g23;  
	A[1][3] = (m*m-1)*g21+(m*m-1)*g22+(2*m*(e+n-1))*g23;
	A[1][4] = (1-m*m)*g21+0.0*g22+(-2*e*m)*g23; 
	A[1][5] = 0.0*g21+(1-m*m)*g22+(-2*m*n)*g23;  
	A[1][6] = (-(m*(m+1))/2.0)*g21+(-(m*(m+1))/2.0)*g22+(-((m+1)*(e+n-1))/2.0-(m*(e+n-1))/2.0)*g23; 
	A[1][7] = ((m*(m+1))/2.0)*g21+0.0*g22+((e*m)/2.0+(e*(m+1))/2.0)*g23;
	A[1][8] = 0.0*g21+((m*(m+1))/2.0)*g22+((m*n)/2.0+(n*(m+1))/2.0)*g23;
	
	A[2][0] = -((m*(m - 1))/2.0)*g31-((m*(m - 1))/2.0)*g32-(((m - 1)*(e + n - 1))/2.0 + (m*(e + n - 1))/2.0)*g33;  
	A[2][1] = -(-(m*(m - 1))/2.0)*g31-0.0*g32-(- (e*m)/2.0 - (e*(m - 1))/2.0)*g33;  
	A[2][2] = -0.0*g31-(-(m*(m - 1))/2.0)*g32-(- (m*n)/2.0 - (n*(m - 1))/2.0)*g33;  
	A[2][3] = (m*m-1)*g31+(m*m-1)*g32+(2*m*(e+n-1))*g33;
	A[2][4] = (1-m*m)*g31+0.0*g32+(-2*e*m)*g33; 
	A[2][5] = 0.0*g31+(1-m*m)*g32+(-2*m*n)*g33;  
	A[2][6] = (-(m*(m+1))/2.0)*g31+(-(m*(m+1))/2.0)*g32+(-((m+1)*(e+n-1))/2.0-(m*(e+n-1))/2.0)*g33; 
	A[2][7] = ((m*(m+1))/2.0)*g31+0.0*g32+((e*m)/2.0+(e*(m+1))/2.0)*g33;
	A[2][8] = 0.0*g31+((m*(m+1))/2.0)*g32+((m*n)/2.0+(n*(m+1))/2.0)*g33;
		
index = ((i*n2p + j)*n3p + k)*36+13;	
	
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][0]*A[0][0]+A[1][0]*A[1][0]+A[2][0]*A[2][0]);
	MC[index+9]=MC[index+9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][1]+A[1][0]*A[1][1]+A[2][0]*A[2][1]);
	MC[index+3]=MC[index+3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][2]+A[1][0]*A[1][2]+A[2][0]*A[2][2]);
	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][3]+A[1][0]*A[1][3]+A[2][0]*A[2][3]);
	MC[index+10]=MC[index+10]+Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][4]+A[1][0]*A[1][4]+A[2][0]*A[2][4]);
	MC[index+4]=MC[index+4] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][5]+A[1][0]*A[1][5]+A[2][0]*A[2][5]);

	MC[index+18]=MC[index+18] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][6]+A[1][0]*A[1][6]+A[2][0]*A[2][6]);
	MC[index+21]=MC[index+21] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][7]+A[1][0]*A[1][7]+A[2][0]*A[2][7]);
	MC[index+19]=MC[index+19] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][8]+A[1][0]*A[1][8]+A[2][0]*A[2][8]);
	
index = (((i+1)*n2p + j)*n3p + k)*36+13;

	MC[index-9]=MC[index-9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][0]+A[1][1]*A[1][0]+A[2][1]*A[2][0]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][1]*A[0][1]+A[1][1]*A[1][1]+A[2][1]*A[2][1]);
	MC[index-6]=MC[index-6] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][2]+A[1][1]*A[1][2]+A[2][1]*A[2][2]);
	MC[index-8]=MC[index-8] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][3]+A[1][1]*A[1][3]+A[2][1]*A[2][3]);
	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][4]+A[1][1]*A[1][4]+A[2][1]*A[2][4]);
	MC[index-5]=MC[index-5] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][5]+A[1][1]*A[1][5]+A[2][1]*A[2][5]);

	MC[index+15]=MC[index+15] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][6]+A[1][1]*A[1][6]+A[2][1]*A[2][6]);
	MC[index+18]=MC[index+18] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][7]+A[1][1]*A[1][7]+A[2][1]*A[2][7]);
	MC[index+16]=MC[index+16] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][8]+A[1][1]*A[1][8]+A[2][1]*A[2][8]);
		
index = ((i*n2p + j+1)*n3p + k)*36+13;

	MC[index-3]=MC[index-3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][0]+A[1][2]*A[1][0]+A[2][2]*A[2][0]);
	MC[index+6]=MC[index+6] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][1]+A[1][2]*A[1][1]+A[2][2]*A[2][1]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][2]*A[0][2]+A[1][2]*A[1][2]+A[2][2]*A[2][2]);
	MC[index-2]=MC[index-2] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][3]+A[1][2]*A[1][3]+A[2][2]*A[2][3]);
	MC[index+7]=MC[index+7] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][4]+A[1][2]*A[1][4]+A[2][2]*A[2][4]);
	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][5]+A[1][2]*A[1][5]+A[2][2]*A[2][5]);
	
	MC[index+17]=MC[index+17] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][6]+A[1][2]*A[1][6]+A[2][2]*A[2][6]);
	MC[index+20]=MC[index+20] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][7]+A[1][2]*A[1][7]+A[2][2]*A[2][7]);
	MC[index+18]=MC[index+18] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][8]+A[1][2]*A[1][8]+A[2][2]*A[2][8]);
		
index = ((i*n2p + j)*n3p + k+1)*36+13;
	
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][0]+A[1][3]*A[1][0]+A[2][3]*A[2][0]);
	MC[index+8]=MC[index+8] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][1]+A[1][3]*A[1][1]+A[2][3]*A[2][1]);
	MC[index+2]=MC[index+2] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][2]+A[1][3]*A[1][2]+A[2][3]*A[2][2]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][3]*A[0][3]+A[1][3]*A[1][3]+A[2][3]*A[2][3]);
	MC[index+9]=MC[index+9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][4]+A[1][3]*A[1][4]+A[2][3]*A[2][4]);
	MC[index+3]=MC[index+3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][5]+A[1][3]*A[1][5]+A[2][3]*A[2][5]);

	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][6]+A[1][3]*A[1][6]+A[2][3]*A[2][6]);
	MC[index+10]=MC[index+10] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][7]+A[1][3]*A[1][7]+A[2][3]*A[2][7]);
	MC[index+4]=MC[index+4] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][8]+A[1][3]*A[1][8]+A[2][3]*A[2][8]);
	
index = (((i+1)*n2p + j)*n3p + k+1)*36+13;
	
	MC[index-10]=MC[index-10]+Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][0]+A[1][4]*A[1][0]+A[2][4]*A[2][0]);
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][1]+A[1][4]*A[1][1]+A[2][4]*A[2][1]);
	MC[index-7]=MC[index-7] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][2]+A[1][4]*A[1][2]+A[2][4]*A[2][2]);
	MC[index-9]=MC[index-9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][3]+A[1][4]*A[1][3]+A[2][4]*A[2][3]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][4]*A[0][4]+A[1][4]*A[1][4]+A[2][4]*A[2][4]);
	MC[index-6]=MC[index-6] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][5]+A[1][4]*A[1][5]+A[2][4]*A[2][5]);

	MC[index-8]=MC[index-8] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][6]+A[1][4]*A[1][6]+A[2][4]*A[2][6]);
	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][7]+A[1][4]*A[1][7]+A[2][4]*A[2][7]);
	MC[index-5]=MC[index-5] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][8]+A[1][4]*A[1][8]+A[2][4]*A[2][8]);

index = ((i*n2p + j+1)*n3p + k+1)*36+13;
	
	MC[index-4]=MC[index-4] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][0]+A[1][5]*A[1][0]+A[2][5]*A[2][0]);
	MC[index+5]=MC[index+5] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][1]+A[1][5]*A[1][1]+A[2][5]*A[2][1]);
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][2]+A[1][5]*A[1][2]+A[2][5]*A[2][2]);
	MC[index-3]=MC[index-3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][3]+A[1][5]*A[1][3]+A[2][5]*A[2][3]);
	MC[index+6]=MC[index+6] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][4]+A[1][5]*A[1][4]+A[2][5]*A[2][4]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][5]*A[0][5]+A[1][5]*A[1][5]+A[2][5]*A[2][5]);
	
	MC[index-2]=MC[index-2] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][6]+A[1][5]*A[1][6]+A[2][5]*A[2][6]);
	MC[index+7]=MC[index+7]  + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][7]+A[1][5]*A[1][7]+A[2][5]*A[2][7]);
	MC[index+1]=MC[index+1]  + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][8]+A[1][5]*A[1][8]+A[2][5]*A[2][8]);
	
				}	

	X1=node[ind+dwe+dsn].x;    X2=node[ind+dsn].x;    X3=node[ind+dwe].x; 	 
    X4=node[ind+dwe+dsn+1].x;  X5=node[ind+dsn+1].x;  X6=node[ind+dwe+1].x; 
   
	Y1=node[ind+dwe+dsn].y;    Y2=node[ind+dsn].y;    Y3=node[ind+dwe].y; 	 
    Y4=node[ind+dwe+dsn+1].y;  Y5=node[ind+dsn+1].y;  Y6=node[ind+dwe+1].y; 
    
	Z1=node[ind+dwe+dsn].z;    Z2=node[ind+dsn].z;    Z3=node[ind+dwe].z; 	 
    Z4=node[ind+dwe+dsn+1].z;  Z5=node[ind+dsn+1].z;  Z6=node[ind+dwe+1].z;
	
	for (ii=0;ii<GaussPoints;ii++)
		for (jj=0;jj<GaussPoints;jj++)
			for (kk=0;kk<GaussPointm;kk++)
			{
				e=Xi[ii];
				n=Xi[jj];
				m=Xim[kk];

	Jacobi[0][0]=-((2.0*m)/(m - 1.0))*X1+ ((2.0*m)/(m - 1.0))*X2+0*X3+((m + 1)/((m - 1.0)))*X4 + (-(m + 1)/((m - 1.0)))*X5+0*X6;
	Jacobi[0][1]=-((2.0*m)/(m - 1.0))*Y1+ ((2.0*m)/(m - 1.0))*Y2+ 0*Y3+ ((m + 1)/((m - 1.0)))*Y4 + (-(m + 1)/((m - 1.0)))*Y5+0*Y6;
	Jacobi[0][2]=-((2.0*m)/(m - 1.0))*Z1+ ((2.0*m)/(m - 1.0))*Z2+ 0*Z3+ ((m + 1)/((m - 1.0)))*Z4 + (-(m + 1)/((m - 1.0)))*Z5+0*Z6;
	Jacobi[1][0]=-((2.0*m)/(m - 1.0))*X1+0*X2+ ((2.0*m)/(m - 1.0))*X3+ ((m + 1)/((m - 1.0)))*X4+0*X5+( -(m + 1)/((m - 1.0)))*X6;
	Jacobi[1][1]=-((2.0*m)/(m - 1.0))*Y1+0*Y2+ ((2.0*m)/(m - 1.0))*Y3+ ((m + 1)/((m - 1.0)))*Y4+0*Y5+( -(m + 1)/((m - 1.0)))*Y6;
	Jacobi[1][2]=-((2.0*m)/(m - 1.0))*Z1+0*Z2+ ((2.0*m)/(m - 1.0))*Z3+ ((m + 1)/((m - 1.0)))*Z4+0*Z5+( -(m + 1)/((m - 1.0)))*Z6;
	Jacobi[2][0]=((2.0*m*(e + n - 1.0))/((m - 1.0)*(m - 1.0)) - (2*(e + n - 1.0))/(m - 1.0))*X1+((2.0*e)/(m - 1.0) - (2.0*e*m)/((m - 1.0)*(m - 1.0)))*X2+((2.0*n)/(m - 1.0) - (2.0*m*n)/((m - 1.0)*(m - 1.0)))*X3+( (e + n - 1.0)/(m - 1.0) - ((m + 1)*(e + n - 1.0))/((m - 1.0)*(m - 1.0)))*X4+((e*(m + 1))/((m - 1.0)*(m - 1.0)) - e/(m - 1.0))*X5+((n*(m + 1))/((m - 1.0)*(m - 1.0)) - n/(m - 1.0))*X6;
	Jacobi[2][1]=((2.0*m*(e + n - 1.0))/((m - 1.0)*(m - 1.0)) - (2*(e + n - 1.0))/(m - 1.0))*Y1+((2.0*e)/(m - 1.0) - (2.0*e*m)/((m - 1.0)*(m - 1.0)))*Y2+((2.0*n)/(m - 1.0) - (2.0*m*n)/((m - 1.0)*(m - 1.0)))*Y3+( (e + n - 1.0)/(m - 1.0) - ((m + 1)*(e + n - 1.0))/((m - 1.0)*(m - 1.0)))*Y4+((e*(m + 1))/((m - 1.0)*(m - 1.0)) - e/(m - 1.0))*Y5+((n*(m + 1))/((m - 1.0)*(m - 1.0)) - n/(m - 1.0))*Y6;
	Jacobi[2][2]=((2.0*m*(e + n - 1.0))/((m - 1.0)*(m - 1.0)) - (2*(e + n - 1.0))/(m - 1.0))*Z1+((2.0*e)/(m - 1.0) - (2.0*e*m)/((m - 1.0)*(m - 1.0)))*Z2+((2.0*n)/(m - 1.0) - (2.0*m*n)/((m - 1.0)*(m - 1.0)))*Z3+( (e + n - 1.0)/(m - 1.0) - ((m + 1)*(e + n - 1.0))/((m - 1.0)*(m - 1.0)))*Z4+((e*(m + 1))/((m - 1.0)*(m - 1.0)) - e/(m - 1.0))*Z5+((n*(m + 1))/((m - 1.0)*(m - 1.0)) - n/(m - 1.0))*Z6;  

	determ = Jacobi[0][0]*((Jacobi[1][1]*Jacobi[2][2]) - (Jacobi[2][1]*Jacobi[1][2])) -Jacobi[0][1]*(Jacobi[1][0]*Jacobi[2][2] - Jacobi[2][0]*Jacobi[1][2]) + Jacobi[0][2]*(Jacobi[1][0]*Jacobi[2][1] - Jacobi[2][0]*Jacobi[1][1]);
	
 	invdet = 1/determ;
	 	
	if ((ii==GaussPoints-1)&&(jj==GaussPoints-1))
    {	e=0.0;    n=0.0;    m=0.0; invdet=0.0; determ=0.0;}

	g11 = (Jacobi[1][1] * Jacobi[2][2] - Jacobi[2][1] * Jacobi[1][2]) * invdet;
	g12 = (Jacobi[0][2] * Jacobi[2][1] - Jacobi[0][1] * Jacobi[2][2]) * invdet;
	g13 = (Jacobi[0][1] * Jacobi[1][2] - Jacobi[0][2] * Jacobi[1][1]) * invdet;
	g21 = (Jacobi[1][2] * Jacobi[2][0] - Jacobi[1][0] * Jacobi[2][2]) * invdet;
	g22 = (Jacobi[0][0] * Jacobi[2][2] - Jacobi[0][2] * Jacobi[2][0]) * invdet;
	g23 = (Jacobi[1][0] * Jacobi[0][2] - Jacobi[0][0] * Jacobi[1][2]) * invdet;
	g31 = (Jacobi[1][0] * Jacobi[2][1] - Jacobi[2][0] * Jacobi[1][1]) * invdet;
	g32 = (Jacobi[2][0] * Jacobi[0][1] - Jacobi[0][0] * Jacobi[2][1]) * invdet;
	g33 = (Jacobi[0][0] * Jacobi[1][1] - Jacobi[1][0] * Jacobi[0][1]) * invdet; 	
		
	A[0][0] = -((m*(m - 1))/2.0)*g11-((m*(m - 1))/2.0)*g12-(((m - 1)*(e + n - 1))/2.0 + (m*(e + n - 1))/2.0)*g13;  
	A[0][1] = -(-(m*(m - 1))/2.0)*g11-0.0*g12-(- (e*m)/2.0 - (e*(m - 1))/2.0)*g13;  
	A[0][2] = -0.0*g11-(-(m*(m - 1))/2.0)*g12-(- (m*n)/2.0 - (n*(m - 1))/2.0)*g13;  
	A[0][3] = (m*m-1)*g11+(m*m-1)*g12+(2*m*(e+n-1))*g13;
	A[0][4] = (1-m*m)*g11+0.0*g12+(-2*e*m)*g13; 
	A[0][5] = 0.0*g11+(1-m*m)*g12+(-2*m*n)*g13;  
	A[0][6] = (-(m*(m+1))/2.0)*g11+(-(m*(m+1))/2.0)*g12+(-((m+1)*(e+n-1))/2.0-(m*(e+n-1))/2.0)*g13; 
	A[0][7] = ((m*(m+1))/2.0)*g11+0.0*g12+((e*m)/2.0+(e*(m+1))/2.0)*g13;
	A[0][8] = 0.0*g11+((m*(m+1))/2.0)*g12+((m*n)/2.0+(n*(m+1))/2.0)*g13; 
	
	A[1][0] = -((m*(m - 1))/2.0)*g21-((m*(m - 1))/2.0)*g22-(((m - 1)*(e + n - 1))/2.0 + (m*(e + n - 1))/2.0)*g23;  
	A[1][1] = -(-(m*(m - 1))/2.0)*g21-0.0*g22-(- (e*m)/2.0 - (e*(m - 1))/2.0)*g23;  
	A[1][2] = -0.0*g21-(-(m*(m - 1))/2.0)*g22-(- (m*n)/2.0 - (n*(m - 1))/2.0)*g23;  
	A[1][3] = (m*m-1)*g21+(m*m-1)*g22+(2*m*(e+n-1))*g23;
	A[1][4] = (1-m*m)*g21+0.0*g22+(-2*e*m)*g23; 
	A[1][5] = 0.0*g21+(1-m*m)*g22+(-2*m*n)*g23;  
	A[1][6] = (-(m*(m+1))/2.0)*g21+(-(m*(m+1))/2.0)*g22+(-((m+1)*(e+n-1))/2.0-(m*(e+n-1))/2.0)*g23; 
	A[1][7] = ((m*(m+1))/2.0)*g21+0.0*g22+((e*m)/2.0+(e*(m+1))/2.0)*g23;
	A[1][8] = 0.0*g21+((m*(m+1))/2.0)*g22+((m*n)/2.0+(n*(m+1))/2.0)*g23;
	
	A[2][0] = -((m*(m - 1))/2.0)*g31-((m*(m - 1))/2.0)*g32-(((m - 1)*(e + n - 1))/2.0 + (m*(e + n - 1))/2.0)*g33;  
	A[2][1] = -(-(m*(m - 1))/2.0)*g31-0.0*g32-(- (e*m)/2.0 - (e*(m - 1))/2.0)*g33;  
	A[2][2] = -0.0*g31-(-(m*(m - 1))/2.0)*g32-(- (m*n)/2.0 - (n*(m - 1))/2.0)*g33;  
	A[2][3] = (m*m-1)*g31+(m*m-1)*g32+(2*m*(e+n-1))*g33;
	A[2][4] = (1-m*m)*g31+0.0*g32+(-2*e*m)*g33; 
	A[2][5] = 0.0*g31+(1-m*m)*g32+(-2*m*n)*g33;  
	A[2][6] = (-(m*(m+1))/2.0)*g31+(-(m*(m+1))/2.0)*g32+(-((m+1)*(e+n-1))/2.0-(m*(e+n-1))/2.0)*g33; 
	A[2][7] = ((m*(m+1))/2.0)*g31+0.0*g32+((e*m)/2.0+(e*(m+1))/2.0)*g33;
	A[2][8] = 0.0*g31+((m*(m+1))/2.0)*g32+((m*n)/2.0+(n*(m+1))/2.0)*g33;
	
/*		if (myid == 6) 	
	{	
	//printf("B %d %d %d|%lf %lf %lf| %.8lf | ",i,j,k,e,n,m,determ);
	printf("%d %d %d|%d %d %d| %lf %lf %lf %lf %lf %lf ",i,j,k,ii,jj,kk,X1,X2,X3,X4,X5,X6);
	printf("| %lf %lf %lf %lf %lf %lf %",Y1,Y2,Y3,Y4,Y5,Y6);
		printf("| %lf %lf %lf %lf %lf %lf ",Z1,Z2,Z3,Z4,Z5,Z6);
for(int kk=0;kk<=8;kk++)
					printf(" %.8lf ",A[0][kk]);
	printf("\n");
	}
*/	
index = (((i+1)*n2p + j+1)*n3p + k)*36+13;
	
	MC[index]   =   MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][0]+A[1][0]*A[1][0]+A[2][0]*A[2][0]);
	MC[index-9] = MC[index-9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][1]+A[1][0]*A[1][1]+A[2][0]*A[2][1]);
	MC[index-3]  =MC[index-3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][2]+A[1][0]*A[1][2]+A[2][0]*A[2][2]);
	MC[index+1] = MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][3]+A[1][0]*A[1][3]+A[2][0]*A[2][3]);
	MC[index-8]  =MC[index-8] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][4]+A[1][0]*A[1][4]+A[2][0]*A[2][4]);
	MC[index-2] = MC[index-2] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][5]+A[1][0]*A[1][5]+A[2][0]*A[2][5]);

	MC[index+18] = MC[index+18] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][6]+A[1][0]*A[1][6]+A[2][0]*A[2][6]);
	MC[index+15] = MC[index-15] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][7]+A[1][0]*A[1][7]+A[2][0]*A[2][7]);
	MC[index+17] = MC[index+17] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][8]+A[1][0]*A[1][8]+A[2][0]*A[2][8]);
		
index = ((i*n2p + j+1)*n3p + k)*36+13;

	MC[index+9]=MC[index+9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][0]+A[1][1]*A[1][0]+A[2][1]*A[2][0]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][1]*A[0][1]+A[1][1]*A[1][1]+A[2][1]*A[2][1]);
	MC[index+6]=MC[index+6] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][2]+A[1][1]*A[1][2]+A[2][1]*A[2][2]);
	MC[index+10]=MC[index+10]+Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][3]+A[1][1]*A[1][3]+A[2][1]*A[2][3]);
	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][4]+A[1][1]*A[1][4]+A[2][1]*A[2][4]);
	MC[index+7]=MC[index+7] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][5]+A[1][1]*A[1][5]+A[2][1]*A[2][5]);
	
	MC[index+21]=MC[index+21] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][6]+A[1][1]*A[1][6]+A[2][1]*A[2][6]);
	MC[index+18]=MC[index+18] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][7]+A[1][1]*A[1][7]+A[2][1]*A[2][7]);
	MC[index+20]=MC[index+20] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][8]+A[1][1]*A[1][8]+A[2][1]*A[2][8]);
		
index = (((i+1)*n2p + j)*n3p + k)*36+13;

	MC[index+3]=MC[index+3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][0]+A[1][2]*A[1][0]+A[2][2]*A[2][0]);
	MC[index-6]=MC[index-6] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][1]+A[1][2]*A[1][1]+A[2][2]*A[2][1]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][2]*A[0][2]+A[1][2]*A[1][2]+A[2][2]*A[2][2]);
	MC[index+4]=MC[index+4] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][3]+A[1][2]*A[1][3]+A[2][2]*A[2][3]);
	MC[index-5]=MC[index-5] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][4]+A[1][2]*A[1][4]+A[2][2]*A[2][4]);
	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][5]+A[1][2]*A[1][5]+A[2][2]*A[2][5]);
	
	MC[index+19]=MC[index+19] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][6]+A[1][2]*A[1][6]+A[2][2]*A[2][6]);
	MC[index+16]=MC[index+16] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][7]+A[1][2]*A[1][7]+A[2][2]*A[2][7]);
	MC[index+18]=MC[index+18] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][8]+A[1][2]*A[1][8]+A[2][2]*A[2][8]);
	

index = (((i+1)*n2p + j+1)*n3p + k+1)*36+13;
	
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][0]+A[1][3]*A[1][0]+A[2][3]*A[2][0]);
	MC[index-10]=MC[index-10]+Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][1]+A[1][3]*A[1][1]+A[2][3]*A[2][1]);
	MC[index-4]=MC[index-4] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][2]+A[1][3]*A[1][2]+A[2][3]*A[2][2]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][3]*A[0][3]+A[1][3]*A[1][3]+A[2][3]*A[2][3]);
	MC[index-9]=MC[index-9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][4]+A[1][3]*A[1][4]+A[2][3]*A[2][4]);
	MC[index-3]=MC[index-3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][5]+A[1][3]*A[1][5]+A[2][3]*A[2][5]);
	
	MC[index+1]=MC[index+1]  + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][6]+A[1][3]*A[1][6]+A[2][3]*A[2][6]);
	MC[index-8]=MC[index-8]  + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][7]+A[1][3]*A[1][7]+A[2][3]*A[2][7]);
	MC[index-2]=MC[index-2]  + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][8]+A[1][3]*A[1][8]+A[2][3]*A[2][8]);
	
index = ((i*n2p + j+1)*n3p + k+1)*36+13;
	
	MC[index+8]=MC[index+8] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][0]+A[1][4]*A[1][0]+A[2][4]*A[2][0]);
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][1]+A[1][4]*A[1][1]+A[2][4]*A[2][1]);
	MC[index+5]=MC[index+5] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][2]+A[1][4]*A[1][2]+A[2][4]*A[2][2]);
	MC[index+9]=MC[index+9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][3]+A[1][4]*A[1][3]+A[2][4]*A[2][3]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][4]*A[0][4]+A[1][4]*A[1][4]+A[2][4]*A[2][4]);
	MC[index+6]=MC[index+6] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][5]+A[1][4]*A[1][5]+A[2][4]*A[2][5]);
	
	MC[index+10]=MC[index+10]  + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][6]+A[1][4]*A[1][6]+A[2][4]*A[2][6]);
	MC[index+1]=MC[index+1]  + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][7]+A[1][4]*A[1][7]+A[2][4]*A[2][7]);
	MC[index+7]=MC[index+7]  + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][8]+A[1][4]*A[1][8]+A[2][4]*A[2][8]);

index = (((i+1)*n2p + j)*n3p + k+1)*36+13;
	
	MC[index+2]=MC[index+2] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][0]+A[1][5]*A[1][0]+A[2][5]*A[2][0]);
	MC[index-7]=MC[index-7] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][1]+A[1][5]*A[1][1]+A[2][5]*A[2][1]);
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][2]+A[1][5]*A[1][2]+A[2][5]*A[2][2]);
	MC[index+3]=MC[index+3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][3]+A[1][5]*A[1][3]+A[2][5]*A[2][3]);
	MC[index-6]=MC[index-6] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][4]+A[1][5]*A[1][4]+A[2][5]*A[2][4]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][5]*A[0][5]+A[1][5]*A[1][5]+A[2][5]*A[2][5]);
	
	MC[index+4]=MC[index+4] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][6]+A[1][5]*A[1][6]+A[2][5]*A[2][6]);
	MC[index-5]=MC[index-5] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][7]+A[1][5]*A[1][7]+A[2][5]*A[2][7]);
	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][8]+A[1][5]*A[1][8]+A[2][5]*A[2][8]);

				}				
				
}
 
void trojB(int64_t i,int64_t j, int64_t k, int64_t ind)
{
	int64_t index;
	int ii,jj,kk;
	
	double e,n,m,determ,invdet;
	double X1,X2,X3,X4,X5,X6,X7,X8;
	double Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8;
	double Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8;
	double g11,g12,g13,g21,g22,g23,g31,g32,g33;
	
	double Jacobi[3][3], A[3][12], K[12][12];
	
	X1=node[ind].x;    X2=node[ind+dwe].x; 	 X3=node[ind+dwe+dsn].x;
    X4=node[ind+1].x;  X5=node[ind+dwe+1].x; X6=node[ind+dwe+dsn+1].x;
   
    Y1=node[ind].y;    Y2=node[ind+dwe].y; 	Y3=node[ind+dwe+dsn].y;
    Y4=node[ind+1].y;    Y5=node[ind+dwe+1].y; 	Y6=node[ind+dwe+dsn+1].y;
    
	Z1=node[ind].z;    Z2=node[ind+dwe].z; 	Z3=node[ind+dwe+dsn].z;
    Z4=node[ind+1].z;    Z5=node[ind+dwe+1].z; 	Z6=node[ind+dwe+dsn+1].z;
								
	for (ii=0;ii<GaussPoints;ii++)
		for (jj=0;jj<GaussPoints;jj++)
			for (kk=0;kk<GaussPointm;kk++)
			{
				e=Xi[ii];
				n=Xi[jj];
				m=Xim[kk];
								
    Jacobi[0][0]=-1/2.0*(1-m)*X1+1/2.0*(1-m)*X2+0*X3+(-1/2.0)*(1+m)*X4+1/2.0*(1+m)*X5+0*X6;
    Jacobi[0][1]=-1/2.0*(1-m)*Y1+1/2.0*(1-m)*Y2+0*Y3+(-1/2.0)*(1+m)*Y4+1/2.0*(1+m)*Y5+0*Y6;
    Jacobi[0][2]=-1/2.0*(1-m)*Z1+1/2.0*(1-m)*Z2+0*Z3+(-1/2.0)*(1+m)*Z4+1/2.0*(1+m)*Z5+0*Z6;
    Jacobi[1][0]= -1/2.0*(1-m)*X1+0*X2+1/2.0*(1-m)*X3+(-1/2.0)*(1+m)*X4+0*X5+1/2.0*(1+m)*X6;
    Jacobi[1][1]= -1/2.0*(1-m)*Y1+0*Y2+1/2.0*(1-m)*Y3+(-1/2.0)*(1+m)*Y4+0*Y5+1/2.0*(1+m)*Y6;
    Jacobi[1][2]= -1/2.0*(1-m)*Z1+0*Z2+1/2.0*(1-m)*Z3+(-1/2.0)*(1+m)*Z4+0*Z5+1/2.0*(1+m)*Z6;
    Jacobi[2][0]= 1/2.0*(-1+e+n)*X1-e*X2/2.0-n*X3/2.0+1/2.0*(1-e-n)*X4+e*X5/2.0+n*X6/2.0;
    Jacobi[2][1]= 1/2.0*(-1+e+n)*Y1-e*Y2/2.0-n*Y3/2.0+1/2.0*(1-e-n)*Y4+e*Y5/2.0+n*Y6/2.0;
    Jacobi[2][2]= 1/2.0*(-1+e+n)*Z1-e*Z2/2.0-n*Z3/2.0+1/2.0*(1-e-n)*Z4+e*Z5/2.0+n*Z6/2.0;   

	determ = Jacobi[0][0]*((Jacobi[1][1]*Jacobi[2][2]) - (Jacobi[2][1]*Jacobi[1][2])) -Jacobi[0][1]*(Jacobi[1][0]*Jacobi[2][2] - Jacobi[2][0]*Jacobi[1][2]) + Jacobi[0][2]*(Jacobi[1][0]*Jacobi[2][1] - Jacobi[2][0]*Jacobi[1][1]);
		
 	invdet = 1/determ;
	
	if ((ii==GaussPoints-1)&&(jj==GaussPoints-1))
    {	e=0.0;    n=0.0;    m=0.0; invdet=0.0; determ=0.0;}
 	
	g11 = (Jacobi[1][1] * Jacobi[2][2] - Jacobi[2][1] * Jacobi[1][2]) * invdet;
	g12 = (Jacobi[0][2] * Jacobi[2][1] - Jacobi[0][1] * Jacobi[2][2]) * invdet;
	g13 = (Jacobi[0][1] * Jacobi[1][2] - Jacobi[0][2] * Jacobi[1][1]) * invdet;
	g21 = (Jacobi[1][2] * Jacobi[2][0] - Jacobi[1][0] * Jacobi[2][2]) * invdet;
	g22 = (Jacobi[0][0] * Jacobi[2][2] - Jacobi[0][2] * Jacobi[2][0]) * invdet;
	g23 = (Jacobi[1][0] * Jacobi[0][2] - Jacobi[0][0] * Jacobi[1][2]) * invdet;
	g31 = (Jacobi[1][0] * Jacobi[2][1] - Jacobi[2][0] * Jacobi[1][1]) * invdet;
	g32 = (Jacobi[2][0] * Jacobi[0][1] - Jacobi[0][0] * Jacobi[2][1]) * invdet;
	g33 = (Jacobi[0][0] * Jacobi[1][1] - Jacobi[1][0] * Jacobi[0][1]) * invdet;
		
	A[0][0] = -1/2.0*(1-m)*g11-1/2.0*(1-m)*g12+1/2.0*(-1+e+n)*g13;  
	A[0][1] = 1/2.0*(1-m)*g11+0.0*g12-e*g13/2.0;  
	A[0][2] = 0.0*g11+1/2.0*(1-m)*g12-n*g13/2.0;  
	A[0][3] = (-1/2.0)*(1+m)*g11+(-1/2.0)*(1+m)*g12+1/2.0*(1-e-n)*g13;
	A[0][4] = 1/2.0*(1+m)*g11+0.0*g12+e*g13/2.0; 
	A[0][5] = 0.0*g11+1/2.0*(1+m)*g12+n*g13/2.0;  
		
	A[1][0] = -1/2.0*(1-m)*g21-1/2.0*(1-m)*g22+1/2.0*(-1+e+n)*g23;
	A[1][1] = 1/2.0*(1-m)*g21+0.0*g22-e*g23/2.0;
	A[1][2] = 0.0*g21+1/2.0*(1-m)*g22-n*g23/2.0;
	A[1][3] = (-1/2.0)*(1+m)*g21+(-1/2.0)*(1+m)*g22+1/2.0*(1-e-n)*g23;
	A[1][4] = 1/2.0*(1+m)*g21+0.0*g22+e*g23/2.0;
	A[1][5] = 0.0*g21+1/2.0*(1+m)*g22+n*g23/2.0;
	
	A[2][0] = -1/2.0*(1-m)*g31-1/2.0*(1-m)*g32+1/2.0*(-1+e+n)*g33;
	A[2][1] = 1/2.0*(1-m)*g31+0.0*g32-e*g33/2.0;
	A[2][2] = 0.0*g31+1/2.0*(1-m)*g32-n*g33/2.0;
	A[2][3] = (-1/2.0)*(1+m)*g31+(-1/2.0)*(1+m)*g32+1/2.0*(1-e-n)*g33;
	A[2][4] = 1/2.0*(1+m)*g31+0.0*g32+e*g33/2.0;
	A[2][5] = 0.0*g31+1/2.0*(1+m)*g32+n*g33/2.0;

index = ((i*n2p + j)*n3p + k)*36+13;	
	
	MC[index] += Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][0]*A[0][0]+A[1][0]*A[1][0]+A[2][0]*A[2][0]);
	MC[index+9] += Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][1]+A[1][0]*A[1][1]+A[2][0]*A[2][1]);
	MC[index+12] += Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][2]+A[1][0]*A[1][2]+A[2][0]*A[2][2]);
	MC[index+1] += Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][3]+A[1][0]*A[1][3]+A[2][0]*A[2][3]);
	MC[index+10] += Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][4]+A[1][0]*A[1][4]+A[2][0]*A[2][4]);
	MC[index+13] += Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][5]+A[1][0]*A[1][5]+A[2][0]*A[2][5]);

index = (((i+1)*n2p + j)*n3p + k)*36+13;

	MC[index-9] += Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][0]+A[1][1]*A[1][0]+A[2][1]*A[2][0]);
	MC[index] += Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][1]*A[0][1]+A[1][1]*A[1][1]+A[2][1]*A[2][1]);
	MC[index+3] += Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][2]+A[1][1]*A[1][2]+A[2][1]*A[2][2]);
	MC[index-8] += Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][3]+A[1][1]*A[1][3]+A[2][1]*A[2][3]);
	MC[index+1] += Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][4]+A[1][1]*A[1][4]+A[2][1]*A[2][4]);
	MC[index+4] += Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][5]+A[1][1]*A[1][5]+A[2][1]*A[2][5]);

index = (((i+1)*n2p + j+1)*n3p + k)*36+13;
	
	MC[index-12] += Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][0]+A[1][2]*A[1][0]+A[2][2]*A[2][0]);
	MC[index-3] += Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][1]+A[1][2]*A[1][1]+A[2][2]*A[2][1]);
	MC[index] += Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][2]+A[1][2]*A[1][2]+A[2][2]*A[2][2]);
	MC[index-11] += Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][3]+A[1][2]*A[1][3]+A[2][2]*A[2][3]);
	MC[index-2] += Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][4]+A[1][2]*A[1][4]+A[2][2]*A[2][4]);
	MC[index+1] += Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][5]+A[1][2]*A[1][5]+A[2][2]*A[2][5]);
	

index = ((i*n2p + j)*n3p + k+1)*36+13;
	
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][0]+A[1][3]*A[1][0]+A[2][3]*A[2][0]);
	MC[index+8]=MC[index+8] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][1]+A[1][3]*A[1][1]+A[2][3]*A[2][1]);
	MC[index+11]=MC[index+11] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][2]+A[1][3]*A[1][2]+A[2][3]*A[2][2]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][3]*A[0][3]+A[1][3]*A[1][3]+A[2][3]*A[2][3]);
	MC[index+9]=MC[index+9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][4]+A[1][3]*A[1][4]+A[2][3]*A[2][4]);
	MC[index+12]=MC[index+12] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][5]+A[1][3]*A[1][5]+A[2][3]*A[2][5]);

index = (((i+1)*n2p + j)*n3p + k+1)*36+13;
	
	MC[index-10]=MC[index-10]+Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][0]+A[1][4]*A[1][0]+A[2][4]*A[2][0]);
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][1]+A[1][4]*A[1][1]+A[2][4]*A[2][1]);
	MC[index+2]=MC[index+2] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][2]+A[1][4]*A[1][2]+A[2][4]*A[2][2]);
	MC[index-9]=MC[index-9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][3]+A[1][4]*A[1][3]+A[2][4]*A[2][3]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][4]*A[0][4]+A[1][4]*A[1][4]+A[2][4]*A[2][4]);
	MC[index+3]=MC[index+3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][5]+A[1][4]*A[1][5]+A[2][4]*A[2][5]);

index = (((i+1)*n2p + j+1)*n3p + k+1)*36+13;
	
	MC[index-13]=MC[index-13] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][0]+A[1][5]*A[1][0]+A[2][5]*A[2][0]);
	MC[index-4]=MC[index-4] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][1]+A[1][5]*A[1][1]+A[2][5]*A[2][1]);
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][2]+A[1][5]*A[1][2]+A[2][5]*A[2][2]);
	MC[index-12]=MC[index-12] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][3]+A[1][5]*A[1][3]+A[2][5]*A[2][3]);
	MC[index-3]=MC[index-3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][4]+A[1][5]*A[1][4]+A[2][5]*A[2][4]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][5]*A[0][5]+A[1][5]*A[1][5]+A[2][5]*A[2][5]);

}						
					
	X1=node[ind+dwe+dsn].x;    X2=node[ind+dsn].x;    X3=node[ind].x; 	 
    X4=node[ind+dwe+dsn+1].x;  X5=node[ind+dsn+1].x;  X6=node[ind+1].x; 
   
	Y1=node[ind+dwe+dsn].y;    Y2=node[ind+dsn].y;    Y3=node[ind].y; 	 
    Y4=node[ind+dwe+dsn+1].y;  Y5=node[ind+dsn+1].y;  Y6=node[ind+1].y; 
    
	Z1=node[ind+dwe+dsn].z;    Z2=node[ind+dsn].z;    Z3=node[ind].z; 	 
    Z4=node[ind+dwe+dsn+1].z;  Z5=node[ind+dsn+1].z;  Z6=node[ind+1].z;
	
	for (ii=0;ii<GaussPoints;ii++)
		for (jj=0;jj<GaussPoints;jj++)
			for (kk=0;kk<GaussPointm;kk++)
			{
				e=Xi[ii];
				n=Xi[jj];
				m=Xim[kk];
 
    Jacobi[0][0]=-1/2.0*(1-m)*X1+1/2.0*(1-m)*X2+0*X3+(-1/2.0)*(1+m)*X4+1/2.0*(1+m)*X5+0*X6;
    Jacobi[0][1]=-1/2.0*(1-m)*Y1+1/2.0*(1-m)*Y2+0*Y3+(-1/2.0)*(1+m)*Y4+1/2.0*(1+m)*Y5+0*Y6;
    Jacobi[0][2]=-1/2.0*(1-m)*Z1+1/2.0*(1-m)*Z2+0*Z3+(-1/2.0)*(1+m)*Z4+1/2.0*(1+m)*Z5+0*Z6;
    Jacobi[1][0]= -1/2.0*(1-m)*X1+0*X2+1/2.0*(1-m)*X3+(-1/2.0)*(1+m)*X4+0*X5+1/2.0*(1+m)*X6;
    Jacobi[1][1]= -1/2.0*(1-m)*Y1+0*Y2+1/2.0*(1-m)*Y3+(-1/2.0)*(1+m)*Y4+0*Y5+1/2.0*(1+m)*Y6;
    Jacobi[1][2]= -1/2.0*(1-m)*Z1+0*Z2+1/2.0*(1-m)*Z3+(-1/2.0)*(1+m)*Z4+0*Z5+1/2.0*(1+m)*Z6;
    Jacobi[2][0]= 1/2.0*(-1+e+n)*X1-e*X2/2.0-n*X3/2.0+1/2.0*(1-e-n)*X4+e*X5/2.0+n*X6/2.0;
    Jacobi[2][1]= 1/2.0*(-1+e+n)*Y1-e*Y2/2.0-n*Y3/2.0+1/2.0*(1-e-n)*Y4+e*Y5/2.0+n*Y6/2.0;
    Jacobi[2][2]= 1/2.0*(-1+e+n)*Z1-e*Z2/2.0-n*Z3/2.0+1/2.0*(1-e-n)*Z4+e*Z5/2.0+n*Z6/2.0;   

	determ = Jacobi[0][0]*((Jacobi[1][1]*Jacobi[2][2]) - (Jacobi[2][1]*Jacobi[1][2])) -Jacobi[0][1]*(Jacobi[1][0]*Jacobi[2][2] - Jacobi[2][0]*Jacobi[1][2]) + Jacobi[0][2]*(Jacobi[1][0]*Jacobi[2][1] - Jacobi[2][0]*Jacobi[1][1]);
		
 	invdet = 1/determ;
	
	if ((ii==GaussPoints-1)&&(jj==GaussPoints-1))
    {	e=0.0;    n=0.0;    m=0.0; invdet=0.0; determ=0.0;}
	 	
	g11 = (Jacobi[1][1] * Jacobi[2][2] - Jacobi[2][1] * Jacobi[1][2]) * invdet;
	g12 = (Jacobi[0][2] * Jacobi[2][1] - Jacobi[0][1] * Jacobi[2][2]) * invdet;
	g13 = (Jacobi[0][1] * Jacobi[1][2] - Jacobi[0][2] * Jacobi[1][1]) * invdet;
	g21 = (Jacobi[1][2] * Jacobi[2][0] - Jacobi[1][0] * Jacobi[2][2]) * invdet;
	g22 = (Jacobi[0][0] * Jacobi[2][2] - Jacobi[0][2] * Jacobi[2][0]) * invdet;
	g23 = (Jacobi[1][0] * Jacobi[0][2] - Jacobi[0][0] * Jacobi[1][2]) * invdet;
	g31 = (Jacobi[1][0] * Jacobi[2][1] - Jacobi[2][0] * Jacobi[1][1]) * invdet;
	g32 = (Jacobi[2][0] * Jacobi[0][1] - Jacobi[0][0] * Jacobi[2][1]) * invdet;
	g33 = (Jacobi[0][0] * Jacobi[1][1] - Jacobi[1][0] * Jacobi[0][1]) * invdet;
		
	A[0][0] = -1/2.0*(1-m)*g11-1/2.0*(1-m)*g12+1/2.0*(-1+e+n)*g13;  
	A[0][1] = 1/2.0*(1-m)*g11+0.0*g12-e*g13/2.0;  
	A[0][2] = 0.0*g11+1/2.0*(1-m)*g12-n*g13/2.0;  
	A[0][3] = (-1/2.0)*(1+m)*g11+(-1/2.0)*(1+m)*g12+1/2.0*(1-e-n)*g13;
	A[0][4] = 1/2.0*(1+m)*g11+0.0*g12+e*g13/2.0; 
	A[0][5] = 0.0*g11+1/2.0*(1+m)*g12+n*g13/2.0; 
	
	A[1][0] = -1/2.0*(1-m)*g21-1/2.0*(1-m)*g22+1/2.0*(-1+e+n)*g23;
	A[1][1] = 1/2.0*(1-m)*g21+0.0*g22-e*g23/2.0;
	A[1][2] = 0.0*g21+1/2.0*(1-m)*g22-n*g23/2.0;
	A[1][3] = (-1/2.0)*(1+m)*g21+(-1/2.0)*(1+m)*g22+1/2.0*(1-e-n)*g23;
	A[1][4] = 1/2.0*(1+m)*g21+0.0*g22+e*g23/2.0;
	A[1][5] = 0.0*g21+1/2.0*(1+m)*g22+n*g23/2.0;
	
	A[2][0] = -1/2.0*(1-m)*g31-1/2.0*(1-m)*g32+1/2.0*(-1+e+n)*g33;
	A[2][1] = 1/2.0*(1-m)*g31+0.0*g32-e*g33/2.0;
	A[2][2] = 0.0*g31+1/2.0*(1-m)*g32-n*g33/2.0;
	A[2][3] = (-1/2.0)*(1+m)*g31+(-1/2.0)*(1+m)*g32+1/2.0*(1-e-n)*g33;
	A[2][4] = 1/2.0*(1+m)*g31+0.0*g32+e*g33/2.0;
	A[2][5] = 0.0*g31+1/2.0*(1+m)*g32+n*g33/2.0;
	
index = (((i+1)*n2p + j+1)*n3p + k)*36+13;
	
	MC[index]   =   MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][0]+A[1][0]*A[1][0]+A[2][0]*A[2][0]);
	MC[index-9] = MC[index-9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][1]+A[1][0]*A[1][1]+A[2][0]*A[2][1]);
	MC[index-12]  =MC[index-12] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][2]+A[1][0]*A[1][2]+A[2][0]*A[2][2]);
	MC[index+1] = MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][3]+A[1][0]*A[1][3]+A[2][0]*A[2][3]);
	MC[index-8]  =MC[index-8] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][4]+A[1][0]*A[1][4]+A[2][0]*A[2][4]);
	MC[index-11] = MC[index-11] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][5]+A[1][0]*A[1][5]+A[2][0]*A[2][5]);

index = ((i*n2p + j+1)*n3p + k)*36+13;

	MC[index+9]=MC[index+9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][0]+A[1][1]*A[1][0]+A[2][1]*A[2][0]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][1]*A[0][1]+A[1][1]*A[1][1]+A[2][1]*A[2][1]);
	MC[index-3]=MC[index-3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][2]+A[1][1]*A[1][2]+A[2][1]*A[2][2]);
	MC[index+10]=MC[index+10]+Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][3]+A[1][1]*A[1][3]+A[2][1]*A[2][3]);
	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][4]+A[1][1]*A[1][4]+A[2][1]*A[2][4]);
	MC[index-2]=MC[index-2] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][5]+A[1][1]*A[1][5]+A[2][1]*A[2][5]);

index = ((i*n2p + j)*n3p + k)*36+13;	
	
	MC[index+12]=MC[index+12] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][0]+A[1][2]*A[1][0]+A[2][2]*A[2][0]);
	MC[index+3]=MC[index+3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][1]+A[1][2]*A[1][1]+A[2][2]*A[2][1]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][2]*A[0][2]+A[1][2]*A[1][2]+A[2][2]*A[2][2]);
	MC[index+13]=MC[index+13] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][3]+A[1][2]*A[1][3]+A[2][2]*A[2][3]);
	MC[index+4]=MC[index+4] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][4]+A[1][2]*A[1][4]+A[2][2]*A[2][4]);
	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][5]+A[1][2]*A[1][5]+A[2][2]*A[2][5]);

index = (((i+1)*n2p + j+1)*n3p + k+1)*36+13;
	
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][0]+A[1][3]*A[1][0]+A[2][3]*A[2][0]);
	MC[index-10]=MC[index-10]+Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][1]+A[1][3]*A[1][1]+A[2][3]*A[2][1]);
	MC[index-13]=MC[index-13] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][2]+A[1][3]*A[1][2]+A[2][3]*A[2][2]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][3]*A[0][3]+A[1][3]*A[1][3]+A[2][3]*A[2][3]);
	MC[index-9]=MC[index-9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][4]+A[1][3]*A[1][4]+A[2][3]*A[2][4]);
	MC[index-12]=MC[index-12] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][5]+A[1][3]*A[1][5]+A[2][3]*A[2][5]);

index = ((i*n2p + j+1)*n3p + k+1)*36+13;
	
	MC[index+8]=MC[index+8] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][0]+A[1][4]*A[1][0]+A[2][4]*A[2][0]);
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][1]+A[1][4]*A[1][1]+A[2][4]*A[2][1]);
	MC[index-4]=MC[index-4] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][2]+A[1][4]*A[1][2]+A[2][4]*A[2][2]);
	MC[index+9]=MC[index+9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][3]+A[1][4]*A[1][3]+A[2][4]*A[2][3]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][4]*A[0][4]+A[1][4]*A[1][4]+A[2][4]*A[2][4]);
	MC[index-3]=MC[index-3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][5]+A[1][4]*A[1][5]+A[2][4]*A[2][5]);
		
index = ((i*n2p + j)*n3p + k+1)*36+13;
	
	MC[index+11]=MC[index+11] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][0]+A[1][5]*A[1][0]+A[2][5]*A[2][0]);
	MC[index+2]=MC[index+2] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][1]+A[1][5]*A[1][1]+A[2][5]*A[2][1]);
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][2]+A[1][5]*A[1][2]+A[2][5]*A[2][2]);
	MC[index+12]=MC[index+12] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][3]+A[1][5]*A[1][3]+A[2][5]*A[2][3]);
	MC[index+3]=MC[index+3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][4]+A[1][5]*A[1][4]+A[2][5]*A[2][4]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][5]*A[0][5]+A[1][5]*A[1][5]+A[2][5]*A[2][5]);
	
				}				
				
}

void trojInfB(int64_t i,int64_t j, int64_t k, int64_t ind)
{
	int64_t index;
	int ii,jj,kk;
	
	double e,n,m,determ,invdet;
	double X1,X2,X3,X4,X5,X6,X7,X8;
	double Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8;
	double Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8;
	double g11,g12,g13,g21,g22,g23,g31,g32,g33;
	
	double Jacobi[3][3], A[3][12], K[12][12];
	
	X1=node[ind].x;    X2=node[ind+dwe].x; 	 X3=node[ind+dwe+dsn].x;
    X4=node[ind+1].x;  X5=node[ind+dwe+1].x; X6=node[ind+dwe+dsn+1].x;
   
    Y1=node[ind].y;    Y2=node[ind+dwe].y; 	Y3=node[ind+dwe+dsn].y;
    Y4=node[ind+1].y;    Y5=node[ind+dwe+1].y; 	Y6=node[ind+dwe+dsn+1].y;
    
	Z1=node[ind].z;    Z2=node[ind+dwe].z; 	Z3=node[ind+dwe+dsn].z;
    Z4=node[ind+1].z;    Z5=node[ind+dwe+1].z; 	Z6=node[ind+dwe+dsn+1].z;
	
		for (ii=0;ii<GaussPoints;ii++)
		for (jj=0;jj<GaussPoints;jj++)
			for (kk=0;kk<GaussPointm;kk++)
			{
				e=Xi[ii];
				n=Xi[jj];
				m=Xim[kk];
								
	Jacobi[0][0]=-((2.0*m)/(m - 1.0))*X1+ ((2.0*m)/(m - 1.0))*X2+0*X3+((m + 1)/((m - 1.0)))*X4 + (-(m + 1)/((m - 1.0)))*X5+0*X6;
	Jacobi[0][1]=-((2.0*m)/(m - 1.0))*Y1+ ((2.0*m)/(m - 1.0))*Y2+ 0*Y3+ ((m + 1)/((m - 1.0)))*Y4 + (-(m + 1)/((m - 1.0)))*Y5+0*Y6;
	Jacobi[0][2]=-((2.0*m)/(m - 1.0))*Z1+ ((2.0*m)/(m - 1.0))*Z2+ 0*Z3+ ((m + 1)/((m - 1.0)))*Z4 + (-(m + 1)/((m - 1.0)))*Z5+0*Z6;
	Jacobi[1][0]=-((2.0*m)/(m - 1.0))*X1+0*X2+ ((2.0*m)/(m - 1.0))*X3+ ((m + 1)/((m - 1.0)))*X4+0*X5+( -(m + 1)/((m - 1.0)))*X6;
	Jacobi[1][1]=-((2.0*m)/(m - 1.0))*Y1+0*Y2+ ((2.0*m)/(m - 1.0))*Y3+ ((m + 1)/((m - 1.0)))*Y4+0*Y5+( -(m + 1)/((m - 1.0)))*Y6;
	Jacobi[1][2]=-((2.0*m)/(m - 1.0))*Z1+0*Z2+ ((2.0*m)/(m - 1.0))*Z3+ ((m + 1)/((m - 1.0)))*Z4+0*Z5+( -(m + 1)/((m - 1.0)))*Z6;
	Jacobi[2][0]=((2.0*m*(e + n - 1.0))/((m - 1.0)*(m - 1.0)) - (2*(e + n - 1.0))/(m - 1.0))*X1+((2.0*e)/(m - 1.0) - (2.0*e*m)/((m - 1.0)*(m - 1.0)))*X2+((2.0*n)/(m - 1.0) - (2.0*m*n)/((m - 1.0)*(m - 1.0)))*X3+( (e + n - 1.0)/(m - 1.0) - ((m + 1)*(e + n - 1.0))/((m - 1.0)*(m - 1.0)))*X4+((e*(m + 1))/((m - 1.0)*(m - 1.0)) - e/(m - 1.0))*X5+((n*(m + 1))/((m - 1.0)*(m - 1.0)) - n/(m - 1.0))*X6;
	Jacobi[2][1]=((2.0*m*(e + n - 1.0))/((m - 1.0)*(m - 1.0)) - (2*(e + n - 1.0))/(m - 1.0))*Y1+((2.0*e)/(m - 1.0) - (2.0*e*m)/((m - 1.0)*(m - 1.0)))*Y2+((2.0*n)/(m - 1.0) - (2.0*m*n)/((m - 1.0)*(m - 1.0)))*Y3+( (e + n - 1.0)/(m - 1.0) - ((m + 1)*(e + n - 1.0))/((m - 1.0)*(m - 1.0)))*Y4+((e*(m + 1))/((m - 1.0)*(m - 1.0)) - e/(m - 1.0))*Y5+((n*(m + 1))/((m - 1.0)*(m - 1.0)) - n/(m - 1.0))*Y6;
	Jacobi[2][2]=((2.0*m*(e + n - 1.0))/((m - 1.0)*(m - 1.0)) - (2*(e + n - 1.0))/(m - 1.0))*Z1+((2.0*e)/(m - 1.0) - (2.0*e*m)/((m - 1.0)*(m - 1.0)))*Z2+((2.0*n)/(m - 1.0) - (2.0*m*n)/((m - 1.0)*(m - 1.0)))*Z3+( (e + n - 1.0)/(m - 1.0) - ((m + 1)*(e + n - 1.0))/((m - 1.0)*(m - 1.0)))*Z4+((e*(m + 1))/((m - 1.0)*(m - 1.0)) - e/(m - 1.0))*Z5+((n*(m + 1))/((m - 1.0)*(m - 1.0)) - n/(m - 1.0))*Z6;  

	determ = Jacobi[0][0]*((Jacobi[1][1]*Jacobi[2][2]) - (Jacobi[2][1]*Jacobi[1][2])) -Jacobi[0][1]*(Jacobi[1][0]*Jacobi[2][2] - Jacobi[2][0]*Jacobi[1][2]) + Jacobi[0][2]*(Jacobi[1][0]*Jacobi[2][1] - Jacobi[2][0]*Jacobi[1][1]);
	
 	invdet = 1/determ;
	 	
	if ((ii==GaussPoints-1)&&(jj==GaussPoints-1))
    {	e=0.0;    n=0.0;    m=0.0; invdet=0.0; determ=0.0;}

	g11 = (Jacobi[1][1] * Jacobi[2][2] - Jacobi[2][1] * Jacobi[1][2]) * invdet;
	g12 = (Jacobi[0][2] * Jacobi[2][1] - Jacobi[0][1] * Jacobi[2][2]) * invdet;
	g13 = (Jacobi[0][1] * Jacobi[1][2] - Jacobi[0][2] * Jacobi[1][1]) * invdet;
	g21 = (Jacobi[1][2] * Jacobi[2][0] - Jacobi[1][0] * Jacobi[2][2]) * invdet;
	g22 = (Jacobi[0][0] * Jacobi[2][2] - Jacobi[0][2] * Jacobi[2][0]) * invdet;
	g23 = (Jacobi[1][0] * Jacobi[0][2] - Jacobi[0][0] * Jacobi[1][2]) * invdet;
	g31 = (Jacobi[1][0] * Jacobi[2][1] - Jacobi[2][0] * Jacobi[1][1]) * invdet;
	g32 = (Jacobi[2][0] * Jacobi[0][1] - Jacobi[0][0] * Jacobi[2][1]) * invdet;
	g33 = (Jacobi[0][0] * Jacobi[1][1] - Jacobi[1][0] * Jacobi[0][1]) * invdet;   	
		
	A[0][0] = -((m*(m - 1))/2.0)*g11-((m*(m - 1))/2.0)*g12-(((m - 1)*(e + n - 1))/2.0 + (m*(e + n - 1))/2.0)*g13;  
	A[0][1] = -(-(m*(m - 1))/2.0)*g11-0.0*g12-(- (e*m)/2.0 - (e*(m - 1))/2.0)*g13;  
	A[0][2] = -0.0*g11-(-(m*(m - 1))/2.0)*g12-(- (m*n)/2.0 - (n*(m - 1))/2.0)*g13;  
	A[0][3] = (m*m-1)*g11+(m*m-1)*g12+(2*m*(e+n-1))*g13;
	A[0][4] = (1-m*m)*g11+0.0*g12+(-2*e*m)*g13; 
	A[0][5] = 0.0*g11+(1-m*m)*g12+(-2*m*n)*g13;  
	A[0][6] = (-(m*(m+1))/2.0)*g11+(-(m*(m+1))/2.0)*g12+(-((m+1)*(e+n-1))/2.0-(m*(e+n-1))/2.0)*g13; 
	A[0][7] = ((m*(m+1))/2.0)*g11+0.0*g12+((e*m)/2.0+(e*(m+1))/2.0)*g13;
	A[0][8] = 0.0*g11+((m*(m+1))/2.0)*g12+((m*n)/2.0+(n*(m+1))/2.0)*g13; 

	A[1][0] = -((m*(m - 1))/2.0)*g21-((m*(m - 1))/2.0)*g22-(((m - 1)*(e + n - 1))/2.0 + (m*(e + n - 1))/2.0)*g23;  
	A[1][1] = -(-(m*(m - 1))/2.0)*g21-0.0*g22-(- (e*m)/2.0 - (e*(m - 1))/2.0)*g23;  
	A[1][2] = -0.0*g21-(-(m*(m - 1))/2.0)*g22-(- (m*n)/2.0 - (n*(m - 1))/2.0)*g23;  
	A[1][3] = (m*m-1)*g21+(m*m-1)*g22+(2*m*(e+n-1))*g23;
	A[1][4] = (1-m*m)*g21+0.0*g22+(-2*e*m)*g23; 
	A[1][5] = 0.0*g21+(1-m*m)*g22+(-2*m*n)*g23;  
	A[1][6] = (-(m*(m+1))/2.0)*g21+(-(m*(m+1))/2.0)*g22+(-((m+1)*(e+n-1))/2.0-(m*(e+n-1))/2.0)*g23; 
	A[1][7] = ((m*(m+1))/2.0)*g21+0.0*g22+((e*m)/2.0+(e*(m+1))/2.0)*g23;
	A[1][8] = 0.0*g21+((m*(m+1))/2.0)*g22+((m*n)/2.0+(n*(m+1))/2.0)*g23;
	
	A[2][0] = -((m*(m - 1))/2.0)*g31-((m*(m - 1))/2.0)*g32-(((m - 1)*(e + n - 1))/2.0 + (m*(e + n - 1))/2.0)*g33;  
	A[2][1] = -(-(m*(m - 1))/2.0)*g31-0.0*g32-(- (e*m)/2.0 - (e*(m - 1))/2.0)*g33;  
	A[2][2] = -0.0*g31-(-(m*(m - 1))/2.0)*g32-(- (m*n)/2.0 - (n*(m - 1))/2.0)*g33;  
	A[2][3] = (m*m-1)*g31+(m*m-1)*g32+(2*m*(e+n-1))*g33;
	A[2][4] = (1-m*m)*g31+0.0*g32+(-2*e*m)*g33; 
	A[2][5] = 0.0*g31+(1-m*m)*g32+(-2*m*n)*g33;  
	A[2][6] = (-(m*(m+1))/2.0)*g31+(-(m*(m+1))/2.0)*g32+(-((m+1)*(e+n-1))/2.0-(m*(e+n-1))/2.0)*g33; 
	A[2][7] = ((m*(m+1))/2.0)*g31+0.0*g32+((e*m)/2.0+(e*(m+1))/2.0)*g33;
	A[2][8] = 0.0*g31+((m*(m+1))/2.0)*g32+((m*n)/2.0+(n*(m+1))/2.0)*g33;
		
index = ((i*n2p + j)*n3p + k)*36+13;	
	
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][0]*A[0][0]+A[1][0]*A[1][0]+A[2][0]*A[2][0]);
	MC[index+9]=MC[index+9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][1]+A[1][0]*A[1][1]+A[2][0]*A[2][1]);
	MC[index+12]=MC[index+12] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][2]+A[1][0]*A[1][2]+A[2][0]*A[2][2]);
	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][3]+A[1][0]*A[1][3]+A[2][0]*A[2][3]);
	MC[index+10]=MC[index+10]+Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][4]+A[1][0]*A[1][4]+A[2][0]*A[2][4]);
	MC[index+13]=MC[index+13] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][5]+A[1][0]*A[1][5]+A[2][0]*A[2][5]);
	
	MC[index+18]=MC[index+18] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][6]+A[1][0]*A[1][6]+A[2][0]*A[2][6]);
	MC[index+21]=MC[index+21] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][7]+A[1][0]*A[1][7]+A[2][0]*A[2][7]);
	MC[index+22]=MC[index+22] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][8]+A[1][0]*A[1][8]+A[2][0]*A[2][8]);
	
index = (((i+1)*n2p + j)*n3p + k)*36+13;

	MC[index-9]=MC[index-9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][0]+A[1][1]*A[1][0]+A[2][1]*A[2][0]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][1]*A[0][1]+A[1][1]*A[1][1]+A[2][1]*A[2][1]);
	MC[index+3]=MC[index+3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][2]+A[1][1]*A[1][2]+A[2][1]*A[2][2]);
	MC[index-8]=MC[index-8] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][3]+A[1][1]*A[1][3]+A[2][1]*A[2][3]);
	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][4]+A[1][1]*A[1][4]+A[2][1]*A[2][4]);
	MC[index+4]=MC[index+4] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][5]+A[1][1]*A[1][5]+A[2][1]*A[2][5]);

	MC[index+15]=MC[index+15] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][6]+A[1][1]*A[1][6]+A[2][1]*A[2][6]);
	MC[index+18]=MC[index+18] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][7]+A[1][1]*A[1][7]+A[2][1]*A[2][7]);
	MC[index+19]=MC[index+19] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][8]+A[1][1]*A[1][8]+A[2][1]*A[2][8]);
		
index = (((i+1)*n2p + j+1)*n3p + k)*36+13;
	
	MC[index-12]=MC[index-12] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][0]+A[1][2]*A[1][0]+A[2][2]*A[2][0]);
	MC[index-3]  =MC[index-3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][1]+A[1][2]*A[1][1]+A[2][2]*A[2][1]);
	MC[index]   =   MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][2]+A[1][2]*A[1][2]+A[2][2]*A[2][2]);
	MC[index-11]=MC[index-11] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][3]+A[1][2]*A[1][3]+A[2][2]*A[2][3]);
	MC[index-2] = MC[index-2] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][4]+A[1][2]*A[1][4]+A[2][2]*A[2][4]);
	MC[index+1] = MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][5]+A[1][2]*A[1][5]+A[2][2]*A[2][5]);
	
	MC[index+14] = MC[index+14] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][6]+A[1][2]*A[1][6]+A[2][2]*A[2][6]);
	MC[index+17] = MC[index+17] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][7]+A[1][2]*A[1][7]+A[2][2]*A[2][7]);
	MC[index+18] = MC[index+18] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][8]+A[1][2]*A[1][8]+A[2][2]*A[2][8]);

index = ((i*n2p + j)*n3p + k+1)*36+13;
	
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][0]+A[1][3]*A[1][0]+A[2][3]*A[2][0]);
	MC[index+8]=MC[index+8] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][1]+A[1][3]*A[1][1]+A[2][3]*A[2][1]);
	MC[index+11]=MC[index+11] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][2]+A[1][3]*A[1][2]+A[2][3]*A[2][2]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][3]*A[0][3]+A[1][3]*A[1][3]+A[2][3]*A[2][3]);
	MC[index+9]=MC[index+9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][4]+A[1][3]*A[1][4]+A[2][3]*A[2][4]);
	MC[index+12]=MC[index+12] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][5]+A[1][3]*A[1][5]+A[2][3]*A[2][5]);

	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][6]+A[1][3]*A[1][6]+A[2][3]*A[2][6]);
	MC[index+10]=MC[index+10] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][7]+A[1][3]*A[1][7]+A[2][3]*A[2][7]);
	MC[index+13]=MC[index+13] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][8]+A[1][3]*A[1][8]+A[2][3]*A[2][8]);
	
index = (((i+1)*n2p + j)*n3p + k+1)*36+13;
	
	MC[index-10]=MC[index-10]+Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][0]+A[1][4]*A[1][0]+A[2][4]*A[2][0]);
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][1]+A[1][4]*A[1][1]+A[2][4]*A[2][1]);
	MC[index+2]=MC[index+2] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][2]+A[1][4]*A[1][2]+A[2][4]*A[2][2]);
	MC[index-9]=MC[index-9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][3]+A[1][4]*A[1][3]+A[2][4]*A[2][3]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][4]*A[0][4]+A[1][4]*A[1][4]+A[2][4]*A[2][4]);
	MC[index+3]=MC[index+3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][5]+A[1][4]*A[1][5]+A[2][4]*A[2][5]);

	MC[index-8]=MC[index-8] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][6]+A[1][4]*A[1][6]+A[2][4]*A[2][6]);
	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][7]+A[1][4]*A[1][7]+A[2][4]*A[2][7]);
	MC[index+4]=MC[index+4] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][8]+A[1][4]*A[1][8]+A[2][4]*A[2][8]);

index = (((i+1)*n2p + j+1)*n3p + k+1)*36+13;
	
	MC[index-13]=MC[index-13] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][0]+A[1][5]*A[1][0]+A[2][5]*A[2][0]);
	MC[index-4]=MC[index-4] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][1]+A[1][5]*A[1][1]+A[2][5]*A[2][1]);
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][2]+A[1][5]*A[1][2]+A[2][5]*A[2][2]);
	MC[index-12]=MC[index-12] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][3]+A[1][5]*A[1][3]+A[2][5]*A[2][3]);
	MC[index-3]=MC[index-3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][4]+A[1][5]*A[1][4]+A[2][5]*A[2][4]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][5]*A[0][5]+A[1][5]*A[1][5]+A[2][5]*A[2][5]);

	MC[index-11]=MC[index-11]  + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][6]+A[1][5]*A[1][6]+A[2][5]*A[2][6]);
	MC[index-2]=MC[index-2]  + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][7]+A[1][5]*A[1][7]+A[2][5]*A[2][7]);
	MC[index+1]=MC[index+1]  + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][8]+A[1][5]*A[1][8]+A[2][5]*A[2][8]);
	}	
				
	X1=node[ind+dwe+dsn].x;    X2=node[ind+dsn].x;    X3=node[ind].x; 	 
    X4=node[ind+dwe+dsn+1].x;  X5=node[ind+dsn+1].x;  X6=node[ind+1].x; 
   
	Y1=node[ind+dwe+dsn].y;    Y2=node[ind+dsn].y;    Y3=node[ind].y; 	 
    Y4=node[ind+dwe+dsn+1].y;  Y5=node[ind+dsn+1].y;  Y6=node[ind+1].y; 
    
	Z1=node[ind+dwe+dsn].z;    Z2=node[ind+dsn].z;    Z3=node[ind].z; 	 
    Z4=node[ind+dwe+dsn+1].z;  Z5=node[ind+dsn+1].z;  Z6=node[ind+1].z;

	for (ii=0;ii<GaussPoints;ii++)
		for (jj=0;jj<GaussPoints;jj++)
			for (kk=0;kk<GaussPointm;kk++)
			{
				e=Xi[ii];
				n=Xi[jj];
				m=Xim[kk];

	Jacobi[0][0]=-((2.0*m)/(m - 1.0))*X1+ ((2.0*m)/(m - 1.0))*X2+0*X3+((m + 1)/((m - 1.0)))*X4 + (-(m + 1)/((m - 1.0)))*X5+0*X6;
	Jacobi[0][1]=-((2.0*m)/(m - 1.0))*Y1+ ((2.0*m)/(m - 1.0))*Y2+ 0*Y3+ ((m + 1)/((m - 1.0)))*Y4 + (-(m + 1)/((m - 1.0)))*Y5+0*Y6;
	Jacobi[0][2]=-((2.0*m)/(m - 1.0))*Z1+ ((2.0*m)/(m - 1.0))*Z2+ 0*Z3+ ((m + 1)/((m - 1.0)))*Z4 + (-(m + 1)/((m - 1.0)))*Z5+0*Z6;
	Jacobi[1][0]=-((2.0*m)/(m - 1.0))*X1+0*X2+ ((2.0*m)/(m - 1.0))*X3+ ((m + 1)/((m - 1.0)))*X4+0*X5+( -(m + 1)/((m - 1.0)))*X6;
	Jacobi[1][1]=-((2.0*m)/(m - 1.0))*Y1+0*Y2+ ((2.0*m)/(m - 1.0))*Y3+ ((m + 1)/((m - 1.0)))*Y4+0*Y5+( -(m + 1)/((m - 1.0)))*Y6;
	Jacobi[1][2]=-((2.0*m)/(m - 1.0))*Z1+0*Z2+ ((2.0*m)/(m - 1.0))*Z3+ ((m + 1)/((m - 1.0)))*Z4+0*Z5+( -(m + 1)/((m - 1.0)))*Z6;
	Jacobi[2][0]=((2.0*m*(e + n - 1.0))/((m - 1.0)*(m - 1.0)) - (2*(e + n - 1.0))/(m - 1.0))*X1+((2.0*e)/(m - 1.0) - (2.0*e*m)/((m - 1.0)*(m - 1.0)))*X2+((2.0*n)/(m - 1.0) - (2.0*m*n)/((m - 1.0)*(m - 1.0)))*X3+( (e + n - 1.0)/(m - 1.0) - ((m + 1)*(e + n - 1.0))/((m - 1.0)*(m - 1.0)))*X4+((e*(m + 1))/((m - 1.0)*(m - 1.0)) - e/(m - 1.0))*X5+((n*(m + 1))/((m - 1.0)*(m - 1.0)) - n/(m - 1.0))*X6;
	Jacobi[2][1]=((2.0*m*(e + n - 1.0))/((m - 1.0)*(m - 1.0)) - (2*(e + n - 1.0))/(m - 1.0))*Y1+((2.0*e)/(m - 1.0) - (2.0*e*m)/((m - 1.0)*(m - 1.0)))*Y2+((2.0*n)/(m - 1.0) - (2.0*m*n)/((m - 1.0)*(m - 1.0)))*Y3+( (e + n - 1.0)/(m - 1.0) - ((m + 1)*(e + n - 1.0))/((m - 1.0)*(m - 1.0)))*Y4+((e*(m + 1))/((m - 1.0)*(m - 1.0)) - e/(m - 1.0))*Y5+((n*(m + 1))/((m - 1.0)*(m - 1.0)) - n/(m - 1.0))*Y6;
	Jacobi[2][2]=((2.0*m*(e + n - 1.0))/((m - 1.0)*(m - 1.0)) - (2*(e + n - 1.0))/(m - 1.0))*Z1+((2.0*e)/(m - 1.0) - (2.0*e*m)/((m - 1.0)*(m - 1.0)))*Z2+((2.0*n)/(m - 1.0) - (2.0*m*n)/((m - 1.0)*(m - 1.0)))*Z3+( (e + n - 1.0)/(m - 1.0) - ((m + 1)*(e + n - 1.0))/((m - 1.0)*(m - 1.0)))*Z4+((e*(m + 1))/((m - 1.0)*(m - 1.0)) - e/(m - 1.0))*Z5+((n*(m + 1))/((m - 1.0)*(m - 1.0)) - n/(m - 1.0))*Z6;  

	determ = Jacobi[0][0]*((Jacobi[1][1]*Jacobi[2][2]) - (Jacobi[2][1]*Jacobi[1][2])) -Jacobi[0][1]*(Jacobi[1][0]*Jacobi[2][2] - Jacobi[2][0]*Jacobi[1][2]) + Jacobi[0][2]*(Jacobi[1][0]*Jacobi[2][1] - Jacobi[2][0]*Jacobi[1][1]);
	
 	invdet = 1/determ;
	 	
	if ((ii==GaussPoints-1)&&(jj==GaussPoints-1))
    {	e=0.0;    n=0.0;    m=0.0; invdet=0.0; determ=0.0;}

	g11 = (Jacobi[1][1] * Jacobi[2][2] - Jacobi[2][1] * Jacobi[1][2]) * invdet;
	g12 = (Jacobi[0][2] * Jacobi[2][1] - Jacobi[0][1] * Jacobi[2][2]) * invdet;
	g13 = (Jacobi[0][1] * Jacobi[1][2] - Jacobi[0][2] * Jacobi[1][1]) * invdet;
	g21 = (Jacobi[1][2] * Jacobi[2][0] - Jacobi[1][0] * Jacobi[2][2]) * invdet;
	g22 = (Jacobi[0][0] * Jacobi[2][2] - Jacobi[0][2] * Jacobi[2][0]) * invdet;
	g23 = (Jacobi[1][0] * Jacobi[0][2] - Jacobi[0][0] * Jacobi[1][2]) * invdet;
	g31 = (Jacobi[1][0] * Jacobi[2][1] - Jacobi[2][0] * Jacobi[1][1]) * invdet;
	g32 = (Jacobi[2][0] * Jacobi[0][1] - Jacobi[0][0] * Jacobi[2][1]) * invdet;
	g33 = (Jacobi[0][0] * Jacobi[1][1] - Jacobi[1][0] * Jacobi[0][1]) * invdet; 	
		
	A[0][0] = -((m*(m - 1))/2.0)*g11-((m*(m - 1))/2.0)*g12-(((m - 1)*(e + n - 1))/2.0 + (m*(e + n - 1))/2.0)*g13;  
	A[0][1] = -(-(m*(m - 1))/2.0)*g11-0.0*g12-(- (e*m)/2.0 - (e*(m - 1))/2.0)*g13;  
	A[0][2] = -0.0*g11-(-(m*(m - 1))/2.0)*g12-(- (m*n)/2.0 - (n*(m - 1))/2.0)*g13;  
	A[0][3] = (m*m-1)*g11+(m*m-1)*g12+(2*m*(e+n-1))*g13;
	A[0][4] = (1-m*m)*g11+0.0*g12+(-2*e*m)*g13; 
	A[0][5] = 0.0*g11+(1-m*m)*g12+(-2*m*n)*g13;  
	A[0][6] = (-(m*(m+1))/2.0)*g11+(-(m*(m+1))/2.0)*g12+(-((m+1)*(e+n-1))/2.0-(m*(e+n-1))/2.0)*g13; 
	A[0][7] = ((m*(m+1))/2.0)*g11+0.0*g12+((e*m)/2.0+(e*(m+1))/2.0)*g13;
	A[0][8] = 0.0*g11+((m*(m+1))/2.0)*g12+((m*n)/2.0+(n*(m+1))/2.0)*g13; 
	
	A[1][0] = -((m*(m - 1))/2.0)*g21-((m*(m - 1))/2.0)*g22-(((m - 1)*(e + n - 1))/2.0 + (m*(e + n - 1))/2.0)*g23;  
	A[1][1] = -(-(m*(m - 1))/2.0)*g21-0.0*g22-(- (e*m)/2.0 - (e*(m - 1))/2.0)*g23;  
	A[1][2] = -0.0*g21-(-(m*(m - 1))/2.0)*g22-(- (m*n)/2.0 - (n*(m - 1))/2.0)*g23;  
	A[1][3] = (m*m-1)*g21+(m*m-1)*g22+(2*m*(e+n-1))*g23;
	A[1][4] = (1-m*m)*g21+0.0*g22+(-2*e*m)*g23; 
	A[1][5] = 0.0*g21+(1-m*m)*g22+(-2*m*n)*g23;  
	A[1][6] = (-(m*(m+1))/2.0)*g21+(-(m*(m+1))/2.0)*g22+(-((m+1)*(e+n-1))/2.0-(m*(e+n-1))/2.0)*g23; 
	A[1][7] = ((m*(m+1))/2.0)*g21+0.0*g22+((e*m)/2.0+(e*(m+1))/2.0)*g23;
	A[1][8] = 0.0*g21+((m*(m+1))/2.0)*g22+((m*n)/2.0+(n*(m+1))/2.0)*g23;
	
	A[2][0] = -((m*(m - 1))/2.0)*g31-((m*(m - 1))/2.0)*g32-(((m - 1)*(e + n - 1))/2.0 + (m*(e + n - 1))/2.0)*g33;  
	A[2][1] = -(-(m*(m - 1))/2.0)*g31-0.0*g32-(- (e*m)/2.0 - (e*(m - 1))/2.0)*g33;  
	A[2][2] = -0.0*g31-(-(m*(m - 1))/2.0)*g32-(- (m*n)/2.0 - (n*(m - 1))/2.0)*g33;  
	A[2][3] = (m*m-1)*g31+(m*m-1)*g32+(2*m*(e+n-1))*g33;
	A[2][4] = (1-m*m)*g31+0.0*g32+(-2*e*m)*g33; 
	A[2][5] = 0.0*g31+(1-m*m)*g32+(-2*m*n)*g33;  
	A[2][6] = (-(m*(m+1))/2.0)*g31+(-(m*(m+1))/2.0)*g32+(-((m+1)*(e+n-1))/2.0-(m*(e+n-1))/2.0)*g33; 
	A[2][7] = ((m*(m+1))/2.0)*g31+0.0*g32+((e*m)/2.0+(e*(m+1))/2.0)*g33;
	A[2][8] = 0.0*g31+((m*(m+1))/2.0)*g32+((m*n)/2.0+(n*(m+1))/2.0)*g33;
	
index = (((i+1)*n2p + j+1)*n3p + k)*36+13;
	
	MC[index]   =   MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][0]+A[1][0]*A[1][0]+A[2][0]*A[2][0]);
	MC[index-9] = MC[index-9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][1]+A[1][0]*A[1][1]+A[2][0]*A[2][1]);
	MC[index-12]  =MC[index-12] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][2]+A[1][0]*A[1][2]+A[2][0]*A[2][2]);
	MC[index+1] = MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][3]+A[1][0]*A[1][3]+A[2][0]*A[2][3]);
	MC[index-8]  =MC[index-8] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][4]+A[1][0]*A[1][4]+A[2][0]*A[2][4]);
	MC[index-11] = MC[index-11] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][5]+A[1][0]*A[1][5]+A[2][0]*A[2][5]);

	MC[index+18] = MC[index+18] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][6]+A[1][0]*A[1][6]+A[2][0]*A[2][6]);
	MC[index+15] = MC[index-15] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][7]+A[1][0]*A[1][7]+A[2][0]*A[2][7]);
	MC[index+14] = MC[index+14] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][0]*A[0][8]+A[1][0]*A[1][8]+A[2][0]*A[2][8]);

index = ((i*n2p + j+1)*n3p + k)*36+13;

	MC[index+9]=MC[index+9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][0]+A[1][1]*A[1][0]+A[2][1]*A[2][0]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][1]*A[0][1]+A[1][1]*A[1][1]+A[2][1]*A[2][1]);
	MC[index-3]=MC[index-3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][2]+A[1][1]*A[1][2]+A[2][1]*A[2][2]);
	MC[index+10]=MC[index+10]+Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][3]+A[1][1]*A[1][3]+A[2][1]*A[2][3]);
	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][4]+A[1][1]*A[1][4]+A[2][1]*A[2][4]);
	MC[index-2]=MC[index-2] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][5]+A[1][1]*A[1][5]+A[2][1]*A[2][5]);	
	
	MC[index+21]=MC[index+21] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][6]+A[1][1]*A[1][6]+A[2][1]*A[2][6]);
	MC[index+18]=MC[index+18] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][7]+A[1][1]*A[1][7]+A[2][1]*A[2][7]);
	MC[index+17]=MC[index+17] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][1]*A[0][8]+A[1][1]*A[1][8]+A[2][1]*A[2][8]);

index = ((i*n2p + j)*n3p + k)*36+13;	
	
	MC[index+12]=MC[index+12] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][0]+A[1][2]*A[1][0]+A[2][2]*A[2][0]);
	MC[index+3]=MC[index+3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][1]+A[1][2]*A[1][1]+A[2][2]*A[2][1]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][2]*A[0][2]+A[1][2]*A[1][2]+A[2][2]*A[2][2]);
	MC[index+13]=MC[index+13] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][3]+A[1][2]*A[1][3]+A[2][2]*A[2][3]);
	MC[index+4]=MC[index+4] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][4]+A[1][2]*A[1][4]+A[2][2]*A[2][4]);
	MC[index+1]=MC[index+1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][5]+A[1][2]*A[1][5]+A[2][2]*A[2][5]);

	MC[index+22]=MC[index+22] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][6]+A[1][2]*A[1][6]+A[2][2]*A[2][6]);
	MC[index+19]=MC[index+19] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][7]+A[1][2]*A[1][7]+A[2][2]*A[2][7]);
	MC[index+18]=MC[index+18] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][2]*A[0][8]+A[1][2]*A[1][8]+A[2][2]*A[2][8]);	

index = (((i+1)*n2p + j+1)*n3p + k+1)*36+13;
	
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][0]+A[1][3]*A[1][0]+A[2][3]*A[2][0]);
	MC[index-10]=MC[index-10]+Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][1]+A[1][3]*A[1][1]+A[2][3]*A[2][1]);
	MC[index-13]=MC[index-13] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][2]+A[1][3]*A[1][2]+A[2][3]*A[2][2]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][3]*A[0][3]+A[1][3]*A[1][3]+A[2][3]*A[2][3]);
	MC[index-9]=MC[index-9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][4]+A[1][3]*A[1][4]+A[2][3]*A[2][4]);
	MC[index-12]=MC[index-12] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][5]+A[1][3]*A[1][5]+A[2][3]*A[2][5]);
	
	MC[index+1]=MC[index+1]  + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][6]+A[1][3]*A[1][6]+A[2][3]*A[2][6]);
	MC[index-8]=MC[index-8]  + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][7]+A[1][3]*A[1][7]+A[2][3]*A[2][7]);
	MC[index-11]=MC[index-11]  + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][3]*A[0][8]+A[1][3]*A[1][8]+A[2][3]*A[2][8]);
	
index = ((i*n2p + j+1)*n3p + k+1)*36+13;
	
	MC[index+8]=MC[index+8] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][0]+A[1][4]*A[1][0]+A[2][4]*A[2][0]);
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][1]+A[1][4]*A[1][1]+A[2][4]*A[2][1]);
	MC[index-4]=MC[index-4] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][2]+A[1][4]*A[1][2]+A[2][4]*A[2][2]);
	MC[index+9]=MC[index+9] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][3]+A[1][4]*A[1][3]+A[2][4]*A[2][3]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][4]*A[0][4]+A[1][4]*A[1][4]+A[2][4]*A[2][4]);
	MC[index-3]=MC[index-3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][5]+A[1][4]*A[1][5]+A[2][4]*A[2][5]);
		
	MC[index+10]=MC[index+10]  + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][6]+A[1][4]*A[1][6]+A[2][4]*A[2][6]);
	MC[index+1]=MC[index+1]  + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][7]+A[1][4]*A[1][7]+A[2][4]*A[2][7]);
	MC[index-2]=MC[index-2]  + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][4]*A[0][8]+A[1][4]*A[1][8]+A[2][4]*A[2][8]);

index = ((i*n2p + j)*n3p + k+1)*36+13;
	
	MC[index+11]=MC[index+11] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][0]+A[1][5]*A[1][0]+A[2][5]*A[2][0]);
	MC[index+2]=MC[index+2] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][1]+A[1][5]*A[1][1]+A[2][5]*A[2][1]);
	MC[index-1]=MC[index-1] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][2]+A[1][5]*A[1][2]+A[2][5]*A[2][2]);
	MC[index+12]=MC[index+12] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][3]+A[1][5]*A[1][3]+A[2][5]*A[2][3]);
	MC[index+3]=MC[index+3] + Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][4]+A[1][5]*A[1][4]+A[2][5]*A[2][4]);
	MC[index]=MC[index] + Wi[ii]*Wi[jj]*Wim[kk]*determ*    (A[0][5]*A[0][5]+A[1][5]*A[1][5]+A[2][5]*A[2][5]);
	
	MC[index+13] += Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][6]+A[1][5]*A[1][6]+A[2][5]*A[2][6]);
	MC[index+4] += Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][7]+A[1][5]*A[1][7]+A[2][5]*A[2][7]);
	MC[index+1] += Wi[ii]*Wi[jj]*Wim[kk]*determ*(A[0][5]*A[0][8]+A[1][5]*A[1][8]+A[2][5]*A[2][8]);

				}				
	
}	
 
void elements()
{
	int64_t i, j, k, ind;

	if (myid == 0) 	
		printf("Compute local element matrices");	
	
//	#pragma omp parallel for private(j,k,ind,index,ii,jj,kk,e,n,m,determ,invdet,Jacobi,A,K,X1,X2,X3,X4,X5,X6,Y1,Y2,Y3,Y4,Y5,Y6,Z1,Z2,Z3,Z4,Z5,Z6,g11,g12,g13,g21,g22,g23,g31,g32,g33)	
	for (i=0;i<=n1;i++)
		for (j=0;j<=n2;j++)
			for (k=0;k<n3-3;k++)
			{
			ind = ((i*n2p+j)*n3p+k);
			
		if (myid < (int)((double)nprocs/4.0))
			{
			if((myid==0 && i==0 && j==1) || (myid==0 && i==1 && j==0))
				{
				trojA(i,j,k,ind);
				}
			else if((myid==((int)((double)nprocs/4.0)-1) && i==n1-1 && j==n2) || (myid==((int)((double)nprocs/4.0)-1) && i==n1 && j==n2-1))
				{
				trojA(i,j,k,ind);
				}
			else
				trojB(i,j,k,ind);
			}
		else if ((myid >= (int)((double)nprocs/4.0)) && (myid < (int)((double)nprocs/2.0)) )
			{
			if(((myid == (int)((double)nprocs/4.0)) && i==0 && j==n2-1) || ((myid == (int)((double)nprocs/4.0)) && i==1 && j==n2))
				{
				trojB(i,j,k,ind);
				}
			else if((myid==((int)((double)nprocs/2.0)-1) && i==n1-1 && j==0) || (myid==((int)((double)nprocs/2.0)-1) && i==n1 && j==1))
				{
				trojB(i,j,k,ind);
				}
			else
				trojA(i,j,k,ind);
			}
			else if ((myid >= (int)((double)nprocs/2.0)) && (myid < (int)(3.0*(double)nprocs/4.0)) )
			{
			if(((myid == (int)((double)nprocs/2.0)) && i==0 && j==1) || ((myid == (int)((double)nprocs/2.0)) && i==1 && j==0))
				{
				trojA(i,j,k,ind);
				}
			else if((myid==((int)(3.0*(double)nprocs/4.0)-1) && i==n1-1 && j==n2) || (myid==((int)(3.0*(double)nprocs/4.0)-1) && i==n1 && j==n2-1))
				{
				trojA(i,j,k,ind);
				}
			else
				trojB(i,j,k,ind);
			}
			else 
			{
			if(((myid == (int)(3.0*(double)nprocs/4.0)) && i==0 && j==n2-1) || ((myid == (int)(3.0*(double)nprocs/4.0)) && i==1 && j==n2))
				{
				trojB(i,j,k,ind);
				}
			else if((myid==(nprocs-1) && i==n1-1 && j==0) || (myid==(nprocs-1) && i==n1 && j==1))
				{
				trojB(i,j,k,ind);
				}
			else
				trojA(i,j,k,ind);
			}				
		
			}
			
	
	if (myid == 0) 	
		printf(" OK ");

// 	#pragma omp parallel for private(j,k,ind,index,ii,jj,kk,e,n,m,determ,invdet,Jacobi,A,K,X1,X2,X3,X4,X5,X6,Y1,Y2,Y3,Y4,Y5,Y6,Z1,Z2,Z3,Z4,Z5,Z6,g11,g12,g13,g21,g22,g23,g31,g32,g33)	
	for (i=0;i<=n1;i++)
		for (j=0;j<=n2;j++)
			{
				k=n3-3;
				ind = ((i*n2p+j)*n3p+k);
	
						if (myid < (int)((double)nprocs/4.0))
			{
			if((myid==0 && i==0 && j==1) || (myid==0 && i==1 && j==0))
				{
				trojInfA(i,j,k,ind);
				}
			else if((myid==((int)((double)nprocs/4.0)-1) && i==n1-1 && j==n2) || (myid==((int)((double)nprocs/4.0)-1) && i==n1 && j==n2-1))
				{
				trojInfA(i,j,k,ind);
				}
			else
				trojInfB(i,j,k,ind);
			}
			else if ((myid >= (int)((double)nprocs/4.0)) && (myid < (int)((double)nprocs/2.0)) )
			{
			if(((myid == (int)((double)nprocs/4.0)) && i==0 && j==n2-1) || ((myid == (int)((double)nprocs/4.0)) && i==1 && j==n2))
				{
				trojInfB(i,j,k,ind);
				}
			else if((myid==((int)((double)nprocs/2.0)-1) && i==n1-1 && j==0) || (myid==((int)((double)nprocs/2.0)-1) && i==n1 && j==1))
				{
				trojInfB(i,j,k,ind);
				}
			else
				trojInfA(i,j,k,ind);
			}
			else if ((myid >= (int)((double)nprocs/2.0)) && (myid < (int)(3.0*(double)nprocs/4.0)) )
			{
			if(((myid == (int)((double)nprocs/2.0)) && i==0 && j==1) || ((myid == (int)((double)nprocs/2.0)) && i==1 && j==0))
				{
				trojInfA(i,j,k,ind);
				}
			else if((myid==((int)(3.0*(double)nprocs/4.0)-1) && i==n1-1 && j==n2) || (myid==((int)(3.0*(double)nprocs/4.0)-1) && i==n1 && j==n2-1))
				{
				trojInfA(i,j,k,ind);
				}
			else
				trojInfB(i,j,k,ind);
			}
			else 
			{
			if(((myid == (int)(3.0*(double)nprocs/4.0)) && i==0 && j==n2-1) || ((myid == (int)(3.0*(double)nprocs/4.0)) && i==1 && j==n2))
				{
				trojInfB(i,j,k,ind);
				}
			else if((myid==(nprocs-1) && i==n1-1 && j==0) || (myid==(nprocs-1) && i==n1 && j==1))
				{
				trojInfB(i,j,k,ind);
				}
			else
				trojInfA(i,j,k,ind);
			}				
		
			}	
		
	if (myid == 0) 	
		printf(" KO \n");
}

double normcross(double *vect_A, double *vect_B)
{
	double cross_P[3];
	double sum=0.0;
		
	cross_P[0] = vect_A[1] * vect_B[2] - vect_A[2] * vect_B[1]; 
    cross_P[1] = vect_A[0] * vect_B[2] - vect_A[2] * vect_B[0]; 
    cross_P[2] = vect_A[0] * vect_B[1] - vect_A[1] * vect_B[0]; 
		
	sum=cross_P[0]*cross_P[0]+cross_P[1]*cross_P[1]+cross_P[2]*cross_P[2];
	return sqrt(sum);  
}

void cross(double *cross_P, double *vect_A, double *vect_B)
{	
	cross_P[0] = vect_A[1] * vect_B[2] - vect_A[2] * vect_B[1]; 
    cross_P[1] =- (vect_A[0] * vect_B[2] - vect_A[2] * vect_B[0]); 
    cross_P[2] = vect_A[0] * vect_B[1] - vect_A[1] * vect_B[0]; 
}

void send(double *x)
{
	int64_t i,k, ind, iglobal;
	double *dTD,*TDw,*dTU,*TUw;
	int quad;

	dTD = (double *)calloc( n1p * n3, sizeof(double));
	TDw = (double *)calloc( n1p * n3, sizeof(double));
	dTU = (double *)calloc( n1p * n3, sizeof(double));
	TUw = (double *)calloc( n1p * n3, sizeof(double));
		
	//----------------------------------------------------------------
	MPI_Isend(&x[(n1-1)*dwe],dwe,MPI_DOUBLE,next,7,MPI_COMM_WORLD,&req1);
	MPI_Isend(&x[2*dwe],dwe,MPI_DOUBLE,prev,7,MPI_COMM_WORLD,&req2);
	MPI_Irecv(&x[(n1+1)*dwe],dwe,MPI_DOUBLE,next,7,MPI_COMM_WORLD,&req3);
	MPI_Irecv(&x[0],dwe,MPI_DOUBLE,prev,7,MPI_COMM_WORLD,&req4);  
			
	MPI_Wait(&req1, &status);
	MPI_Wait(&req2, &status);
	MPI_Wait(&req3, &status);
	MPI_Wait(&req4, &status);

	quad = nprocs - 1 - myid;
		
	for (i = 0; i <= n1+1; i++)
		for (k=0;k<n3;k++)	
		{
			ind = ((i*n2p+2)*n3p+k);
			
			iglobal= ((i)*n3+k);	
			dTD[iglobal] = x[ind];
		}
		
	MPI_Isend(dTD,n1p*n3,MPI_DOUBLE,quad,7,MPI_COMM_WORLD,&req1);
	MPI_Irecv(TDw,n1p*n3,MPI_DOUBLE,quad,7,MPI_COMM_WORLD,&req2);
	MPI_Wait(&req1, &status);
	MPI_Wait(&req2, &status);
		
	for (i = 0; i <= n1+1; i++)
		for (k=0;k<n3;k++)	
		{
			ind = ((i*n2p+0)*n3p+k);
			iglobal= ((n1+1-i)*n3+k);	
			x[ind]= TDw[iglobal];
		}

	for (i = 0; i <= n1+1; i++)
		for (k=0;k<n3;k++)	
		{
			ind = ((i*n2p+n2-1)*n3p+k);
			iglobal= (i*n3+k);
			
			dTU[iglobal] = x[ind];
		}

	if (myid < (int)((double)nprocs/2.0))
		quad = ((double)nprocs/2.0) - 1 - myid;
	else
		quad = nprocs - 1 + (int)((double)nprocs/2.0)  - myid;	

	MPI_Isend(dTU,n1p*n3,MPI_DOUBLE,quad,7,MPI_COMM_WORLD,&req3);
	MPI_Irecv(TUw,n1p*n3,MPI_DOUBLE,quad,7,MPI_COMM_WORLD,&req4);	
	MPI_Wait(&req3, &status);
	MPI_Wait(&req4, &status);
	
	for (i = 0; i <= n1; i++)
		for (k=0;k<n3;k++)	
		{
			ind = ((i*n2p+n2+1)*n3p+k);
			iglobal= ((n1+1-i)*n3+k);	
			x[ind]= TUw[iglobal];
		}
				
	free(dTD);	
	free(TDw);	
	free(dTU);	
	free(TUw);
		
	MPI_Isend(&x[(n1-1)*dwe],dwe,MPI_DOUBLE,next,7,MPI_COMM_WORLD,&req1);
	MPI_Isend(&x[2*dwe],dwe,MPI_DOUBLE,prev,7,MPI_COMM_WORLD,&req2);
	MPI_Irecv(&x[(n1+1)*dwe],dwe,MPI_DOUBLE,next,7,MPI_COMM_WORLD,&req3);
	MPI_Irecv(&x[0],dwe,MPI_DOUBLE,prev,7,MPI_COMM_WORLD,&req4);  
			
	MPI_Wait(&req1, &status);
	MPI_Wait(&req2, &status);
	MPI_Wait(&req3, &status);
	MPI_Wait(&req4, &status);

}

double normvekt(double *vect_A)
{
	double sum=0.0;
				
	sum=vect_A[0]*vect_A[0]+vect_A[1]*vect_A[1]+vect_A[2]*vect_A[2];
	return sqrt(sum);  
}

void trojBcA(int64_t i,int64_t j, int64_t ind, int64_t iglobal, double *dg) 
{
	int64_t ij, index;
	double normal[3],tangent1[3],tangent2[3], svektor[3],node_mean[3];
	double face,tmp,dltang1,dltang2,alpha,beta,gamma;
			
			tangent1[0]=node[ind+dwe].x-node[ind].x;
			tangent1[1]=node[ind+dwe].y-node[ind].y;
			tangent1[2]=node[ind+dwe].z-node[ind].z;
			
			tangent2[0]=node[ind+dsn].x-node[ind].x;
			tangent2[1]=node[ind+dsn].y-node[ind].y;
			tangent2[2]=node[ind+dsn].z-node[ind].z;
						
			face=(normcross(tangent1,tangent2)/2.0)/3.0;
			
			cross(normal,tangent2,tangent1);
			
			tmp=normvekt(normal);
			for(ij=0;ij<3;ij++)
				normal[ij]=normal[ij]/tmp;			
			dltang1=normvekt(tangent1);	
	
			for(ij=0;ij<3;ij++)
				tangent1[ij]=tangent1[ij]/dltang1;
						
			dltang2=normvekt(tangent2);	
			for(ij=0;ij<3;ij++)
				tangent2[ij]=tangent2[ij]/dltang2;
			
			node_mean[0]=node[ind].x;
			node_mean[1]=node[ind].y;
			node_mean[2]=node[ind].z;
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=0.0-node_mean[ij];
			
			tmp=normvekt(svektor);
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=svektor[ij]/tmp;
										
			alpha=normal[0]*svektor[0]+normal[1]*svektor[1]+normal[2]*svektor[2];
			beta=tangent1[0]*svektor[0]+tangent1[1]*svektor[1]+tangent1[2]*svektor[2];
			gamma=tangent2[0]*svektor[0]+tangent2[1]*svektor[1]+tangent2[2]*svektor[2];
						
			b[ind]+= (dg[iglobal]/alpha) * face;

			index = ((i*n2p + j)*n3p)*36+13;	
						
			MC[index]+= (beta/alpha)*face*(-1.0/dltang1) + (gamma/alpha)*face*(-1.0/dltang2);
			MC[index+9]+= (beta/alpha)*face*(1.0/dltang1) + (gamma/alpha)*face*(0.0/dltang2);
			MC[index+3]+= (beta/alpha)*face*(0.0/dltang1) + (gamma/alpha)*face*(1.0/dltang2);
			
			node_mean[0]=node[ind+dwe].x;
			node_mean[1]=node[ind+dwe].y;
			node_mean[2]=node[ind+dwe].z;
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=0.0-node_mean[ij];
			
			tmp=normvekt(svektor);
						
			for(ij=0;ij<3;ij++)
				svektor[ij]=svektor[ij]/tmp;
			
			alpha=normal[0]*svektor[0]+normal[1]*svektor[1]+normal[2]*svektor[2];
			beta=tangent1[0]*svektor[0]+tangent1[1]*svektor[1]+tangent1[2]*svektor[2];
			gamma=tangent2[0]*svektor[0]+tangent2[1]*svektor[1]+tangent2[2]*svektor[2];

			b[ind+dwe]+= (dg[iglobal+n2p*(1)+0]/alpha) * face;
			
			index = (((i+1)*n2p + j)*n3p )*36+13;
			
			MC[index-9]+= (beta/alpha)*face*(-1.0/dltang1) + (gamma/alpha)*face*(-1.0/dltang2);
			MC[index]+= (beta/alpha)*face*(1.0/dltang1) + (gamma/alpha)*face*(0.0/dltang2);
			MC[index-6]+= (beta/alpha)*face*(0.0/dltang1) + (gamma/alpha)*face*(1.0/dltang2);
			
			node_mean[0]=node[ind+dsn].x;
			node_mean[1]=node[ind+dsn].y;
			node_mean[2]=node[ind+dsn].z;
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=0.0-node_mean[ij];
			
			tmp=normvekt(svektor);
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=svektor[ij]/tmp;
			
			alpha=normal[0]*svektor[0]+normal[1]*svektor[1]+normal[2]*svektor[2];
			beta=tangent1[0]*svektor[0]+tangent1[1]*svektor[1]+tangent1[2]*svektor[2];
			gamma=tangent2[0]*svektor[0]+tangent2[1]*svektor[1]+tangent2[2]*svektor[2];
			
			b[ind+dsn]+= (dg[iglobal+n2p*0+1]/alpha) * face;
						
			index = ((i*n2p + j+1)*n3p )*36+13;
			
			MC[index-3]+= (beta/alpha)*face*(-1.0/dltang1) + (gamma/alpha)*face*(-1.0/dltang2);
			MC[index+6]+= (beta/alpha)*face*(1.0/dltang1) + (gamma/alpha)*face*(0.0/dltang2);
			MC[index]+= (beta/alpha)*face*(0.0/dltang1) + (gamma/alpha)*face*(1.0/dltang2);
		
			tangent2[0]=node[ind+dwe].x-node[ind+dwe+dsn].x;
			tangent2[1]=node[ind+dwe].y-node[ind+dwe+dsn].y;
			tangent2[2]=node[ind+dwe].z-node[ind+dwe+dsn].z;
			
			tangent1[0]=node[ind+dsn].x-node[ind+dwe+dsn].x;
			tangent1[1]=node[ind+dsn].y-node[ind+dwe+dsn].y;
			tangent1[2]=node[ind+dsn].z-node[ind+dwe+dsn].z;
						
			face=(normcross(tangent1,tangent2)/2.0)/3.0;
			
			cross(normal,tangent2,tangent1);
			
			tmp=normvekt(normal);
			for(ij=0;ij<3;ij++)
				normal[ij]=normal[ij]/tmp;
			
			dltang1=normvekt(tangent1);	
			for(ij=0;ij<3;ij++)
				tangent1[ij]=tangent1[ij]/dltang1;
						
			dltang2=normvekt(tangent2);		
			for(ij=0;ij<3;ij++)
				tangent2[ij]=tangent2[ij]/dltang2;
			
			node_mean[0]=node[ind+dwe+dsn].x;
			node_mean[1]=node[ind+dwe+dsn].y;
			node_mean[2]=node[ind+dwe+dsn].z;
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=0.0-node_mean[ij];
			
			tmp=normvekt(svektor);
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=svektor[ij]/tmp;
											
			alpha=normal[0]*svektor[0]+normal[1]*svektor[1]+normal[2]*svektor[2];
			beta=tangent1[0]*svektor[0]+tangent1[1]*svektor[1]+tangent1[2]*svektor[2];
			gamma=tangent2[0]*svektor[0]+tangent2[1]*svektor[1]+tangent2[2]*svektor[2];
			
			b[ind+dwe+dsn]+= (dg[iglobal+n2p*(1)+1]/alpha) * face;

			index = (((i+1)*n2p + j+1)*n3p )*36+13;
						
			MC[index]+= (beta/alpha)*face*(-1.0/dltang1) + (gamma/alpha)*face*(-1.0/dltang2);
			MC[index-9]+= (beta/alpha)*face*(1.0/dltang1) + (gamma/alpha)*face*(0.0/dltang2);
			MC[index-3]+= (beta/alpha)*face*(0.0/dltang1) + (gamma/alpha)*face*(1.0/dltang2);

			node_mean[0]=node[ind+dsn].x;
			node_mean[1]=node[ind+dsn].y;
			node_mean[2]=node[ind+dsn].z;
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=0.0-node_mean[ij];
			
			tmp=normvekt(svektor);
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=svektor[ij]/tmp;
										
			alpha=normal[0]*svektor[0]+normal[1]*svektor[1]+normal[2]*svektor[2];
			beta=tangent1[0]*svektor[0]+tangent1[1]*svektor[1]+tangent1[2]*svektor[2];
			gamma=tangent2[0]*svektor[0]+tangent2[1]*svektor[1]+tangent2[2]*svektor[2];
			
			b[ind+dsn]+= (dg[iglobal+n2p*0+1]/alpha) * face;
						
			index = ((i*n2p + j+1)*n3p )*36+13;
			
			MC[index+9]+= (beta/alpha)*face*(-1.0/dltang1) + (gamma/alpha)*face*(-1.0/dltang2);
			MC[index]+= (beta/alpha)*face*(1.0/dltang1) + (gamma/alpha)*face*(0.0/dltang2);
			MC[index+6]+= (beta/alpha)*face*(0.0/dltang1) + (gamma/alpha)*face*(1.0/dltang2);

			node_mean[0]=node[ind+dwe].x;
			node_mean[1]=node[ind+dwe].y;
			node_mean[2]=node[ind+dwe].z;
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=0.0-node_mean[ij];
			
			tmp=normvekt(svektor);
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=svektor[ij]/tmp;
	
			alpha=normal[0]*svektor[0]+normal[1]*svektor[1]+normal[2]*svektor[2];
			beta=tangent1[0]*svektor[0]+tangent1[1]*svektor[1]+tangent1[2]*svektor[2];
			gamma=tangent2[0]*svektor[0]+tangent2[1]*svektor[1]+tangent2[2]*svektor[2];			
			
			b[ind+dwe]+= (dg[iglobal+n2p*(1)+0]/alpha) * face;
			
			index = (((i+1)*n2p + j)*n3p )*36+13;
			
			MC[index+3]+= (beta/alpha)*face*(-1.0/dltang1) + (gamma/alpha)*face*(-1.0/dltang2);
			MC[index-6]+= (beta/alpha)*face*(1.0/dltang1) + (gamma/alpha)*face*(0.0/dltang2);
			MC[index]+= (beta/alpha)*face*(0.0/dltang1) + (gamma/alpha)*face*(1.0/dltang2);
}

void trojBcB(int64_t i,int64_t j, int64_t ind, int64_t iglobal, double *dg) 
{
	int64_t ij, index;
	double dlXb[3],dlXu[3],dlYl[3],dlYr[3],normal[3],tangent1[3],tangent2[3],svektor[3],node_mean[3];
	double face,tmp,dltang1,dltang2,alpha,beta,gamma;
					
			tangent1[0]=node[ind].x-node[ind+dwe].x;
			tangent1[1]=node[ind].y-node[ind+dwe].y;
			tangent1[2]=node[ind].z-node[ind+dwe].z;
			
			tangent2[0]=node[ind+dwe+dsn].x-node[ind+dwe].x;
			tangent2[1]=node[ind+dwe+dsn].y-node[ind+dwe].y;
			tangent2[2]=node[ind+dwe+dsn].z-node[ind+dwe].z;
						
			face=(normcross(tangent1,tangent2)/2.0)/3.0;
			
			cross(normal,tangent1,tangent2);
			
			tmp=normvekt(normal);
			for(ij=0;ij<3;ij++)
				normal[ij]=normal[ij]/tmp;
			
			dltang1=normvekt(tangent1);	
			for(ij=0;ij<3;ij++)
				tangent1[ij]=tangent1[ij]/dltang1;
						
			dltang2=normvekt(tangent2);		
			for(ij=0;ij<3;ij++)
				tangent2[ij]=tangent2[ij]/dltang2;
	
			node_mean[0]=node[ind].x;
			node_mean[1]=node[ind].y;
			node_mean[2]=node[ind].z;
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=0.0-node_mean[ij];
			
			tmp=normvekt(svektor);
			for(ij=0;ij<3;ij++)
				svektor[ij]=svektor[ij]/tmp;
										
			alpha=normal[0]*svektor[0]+normal[1]*svektor[1]+normal[2]*svektor[2];
			beta=tangent1[0]*svektor[0]+tangent1[1]*svektor[1]+tangent1[2]*svektor[2];
			gamma=tangent2[0]*svektor[0]+tangent2[1]*svektor[1]+tangent2[2]*svektor[2];
						
			b[ind]+= (dg[iglobal]/alpha) * face;
			
			index = ((i*n2p + j)*n3p)*36+13;	
						
			MC[index]+= (beta/alpha)*face*(1.0/dltang1) + (gamma/alpha)*face*(0.0/dltang2);
			MC[index+9]+= (beta/alpha)*face*(-1.0/dltang1) + (gamma/alpha)*face*(-1.0/dltang2);
			MC[index+12]+= (beta/alpha)*face*(0.0/dltang1) + (gamma/alpha)*face*(1.0/dltang2);
		
			node_mean[0]=node[ind+dwe].x;
			node_mean[1]=node[ind+dwe].y;
			node_mean[2]=node[ind+dwe].z;
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=0.0-node_mean[ij];
			
			tmp=normvekt(svektor);
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=svektor[ij]/tmp;
			
			alpha=normal[0]*svektor[0]+normal[1]*svektor[1]+normal[2]*svektor[2];
			beta=tangent1[0]*svektor[0]+tangent1[1]*svektor[1]+tangent1[2]*svektor[2];
			gamma=tangent2[0]*svektor[0]+tangent2[1]*svektor[1]+tangent2[2]*svektor[2];

			b[ind+dwe]+= (dg[iglobal+n2p*(1)+0]/alpha) * face;
			
			index = (((i+1)*n2p + j)*n3p )*36+13;
			
			MC[index-9]+= (beta/alpha)*face*(1.0/dltang1) + (gamma/alpha)*face*(0.0/dltang2);
			MC[index]+= (beta/alpha)*face*(-1.0/dltang1) + (gamma/alpha)*face*(-1.0/dltang2);
			MC[index+3]+= (beta/alpha)*face*(0.0/dltang1) + (gamma/alpha)*face*(1.0/dltang2);

			node_mean[0]=node[ind+dwe+dsn].x;
			node_mean[1]=node[ind+dwe+dsn].y;
			node_mean[2]=node[ind+dwe+dsn].z;
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=0.0-node_mean[ij];
			
			tmp=normvekt(svektor);
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=svektor[ij]/tmp;
			
			alpha=normal[0]*svektor[0]+normal[1]*svektor[1]+normal[2]*svektor[2];
			beta=tangent1[0]*svektor[0]+tangent1[1]*svektor[1]+tangent1[2]*svektor[2];
			gamma=tangent2[0]*svektor[0]+tangent2[1]*svektor[1]+tangent2[2]*svektor[2];
						
			b[ind+dwe+dsn]+= (dg[iglobal+n2p*1+1]/alpha) * face;
						
			index = (((i+1)*n2p + j+1)*n3p)*36+13;
			
			MC[index-12]+= (beta/alpha)*face*(1.0/dltang1) + (gamma/alpha)*face*(0.0/dltang2);
			MC[index-3]+= (beta/alpha)*face*(-1.0/dltang1) + (gamma/alpha)*face*(-1.0/dltang2);
			MC[index]+= (beta/alpha)*face*(0.0/dltang1) + (gamma/alpha)*face*(1.0/dltang2);

			//Second triangle			
			tangent1[0]=node[ind+dwe+dsn].x-node[ind+dsn].x;
			tangent1[1]=node[ind+dwe+dsn].y-node[ind+dsn].y;
			tangent1[2]=node[ind+dwe+dsn].z-node[ind+dsn].z;
			
			tangent2[0]=node[ind].x-node[ind+dsn].x;
			tangent2[1]=node[ind].y-node[ind+dsn].y;
			tangent2[2]=node[ind].z-node[ind+dsn].z;
			
			face=(normcross(tangent1,tangent2)/2.0)/3.0;

			cross(normal,tangent1,tangent2);
			
			tmp=normvekt(normal);
			for(ij=0;ij<3;ij++)
				normal[ij]=normal[ij]/tmp;
			
			dltang1=normvekt(tangent1);	
			for(ij=0;ij<3;ij++)
				tangent1[ij]=tangent1[ij]/dltang1;
			
			dltang2=normvekt(tangent2);	
			for(ij=0;ij<3;ij++)
				tangent2[ij]=tangent2[ij]/dltang2;
	
			node_mean[0]=node[ind+dwe+dsn].x;
			node_mean[1]=node[ind+dwe+dsn].y;
			node_mean[2]=node[ind+dwe+dsn].z;
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=0.0-node_mean[ij];
			
			tmp=normvekt(svektor);
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=svektor[ij]/tmp;
											
			alpha=normal[0]*svektor[0]+normal[1]*svektor[1]+normal[2]*svektor[2];
			beta=tangent1[0]*svektor[0]+tangent1[1]*svektor[1]+tangent1[2]*svektor[2];
			gamma=tangent2[0]*svektor[0]+tangent2[1]*svektor[1]+tangent2[2]*svektor[2];
			
			b[ind+dwe+dsn]+= (dg[iglobal+n2p*(1)+1]/alpha) * face;
			
			index = (((i+1)*n2p + j+1)*n3p )*36+13;
						
			MC[index]+= (beta/alpha)*face*(1.0/dltang1) + (gamma/alpha)*face*(0.0/dltang2);
			MC[index-9]+= (beta/alpha)*face*(-1.0/dltang1) + (gamma/alpha)*face*(-1.0/dltang2);
			MC[index-12]+= (beta/alpha)*face*(0.0/dltang1) + (gamma/alpha)*face*(1.0/dltang2);

			node_mean[0]=node[ind+dsn].x;
			node_mean[1]=node[ind+dsn].y;
			node_mean[2]=node[ind+dsn].z;
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=0.0-node_mean[ij];
			
			tmp=normvekt(svektor);
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=svektor[ij]/tmp;
										
			alpha=normal[0]*svektor[0]+normal[1]*svektor[1]+normal[2]*svektor[2];
			beta=tangent1[0]*svektor[0]+tangent1[1]*svektor[1]+tangent1[2]*svektor[2];
			gamma=tangent2[0]*svektor[0]+tangent2[1]*svektor[1]+tangent2[2]*svektor[2];
			
			b[ind+dsn]+= (dg[iglobal+n2p*0+1]/alpha) * face;

			index = ((i*n2p + j+1)*n3p )*36+13;
			
			MC[index+9]+= (beta/alpha)*face*(1.0/dltang1) + (gamma/alpha)*face*(0.0/dltang2);
			MC[index]+= (beta/alpha)*face*(-1.0/dltang1) + (gamma/alpha)*face*(-1.0/dltang2);
			MC[index-3]+= (beta/alpha)*face*(0.0/dltang1) + (gamma/alpha)*face*(1.0/dltang2);

			node_mean[0]=node[ind].x;
			node_mean[1]=node[ind].y;
			node_mean[2]=node[ind].z;
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=0.0-node_mean[ij];
			
			tmp=normvekt(svektor);
			
			for(ij=0;ij<3;ij++)
				svektor[ij]=svektor[ij]/tmp;
	
			alpha=normal[0]*svektor[0]+normal[1]*svektor[1]+normal[2]*svektor[2];
			beta=tangent1[0]*svektor[0]+tangent1[1]*svektor[1]+tangent1[2]*svektor[2];
			gamma=tangent2[0]*svektor[0]+tangent2[1]*svektor[1]+tangent2[2]*svektor[2];			
			
			b[ind]+= (dg[iglobal+n2p*(0)+0]/alpha) * face;
			
			index = ((i*n2p + j)*n3p)*36+13;

			MC[index+12]+= (beta/alpha)*face*(1.0/dltang1) + (gamma/alpha)*face*(0.0/dltang2);
			MC[index+3]+= (beta/alpha)*face*(-1.0/dltang1) + (gamma/alpha)*face*(-1.0/dltang2);
			MC[index]+= (beta/alpha)*face*(0.0/dltang1) + (gamma/alpha)*face*(1.0/dltang2);
}

void associate_dg_BC()
{
	int64_t i, j,k, ind, iglobal, index;
	
	double *dg;
	
	dg = (double *)calloc(N1 * n2p, sizeof(double));
		
	if(myid==0)
	{	
	printf("Reading dg BC");
	fr = fopen(OPEN_file_dg,"r");
	
	for (j=1;j<=n2;j++)
		for (i=0;i<N1;i++)	
		
		{
			ind = (i*n2p+j);
			fscanf(fr, "%lf", &tmp);
			dg[ind]= tmp * 0.00001;
		}		
	
	fclose(fr);
	
	for (i = 0; i < N1; i++)
		dg[(N1-i-1)*n2p+0]=dg[i*n2p+2];					
						
	for (i = 0; i < N1; i++)
		{
		if ( i<=(int)((double)Ne1/2.0))
			{
				dg[((int)((double)Ne1/2.0)-i)*n2p+n2+1]=dg[i*n2p+n2-1];	
			}
			else 
			{
				dg[(N1-i-1+(int)((double)Ne1/2.0))*n2p+n2+1]=dg[i*n2p+n2-1];
			}
		}
		dg[(N1-1)*n2p+n2+1]=dg[0*n2p+n2+1];
		
	printf(" OK dg\n");		
	}
		
	MPI_Bcast(dg, N1 * n2p, MPI_DOUBLE,0, MPI_COMM_WORLD);	

	if(myid==0)	printf("Compute dg BC ");	
	
	for (j=0;j<=n2;j++)
		for (i=0;i<=n1;i++)
			{
			ind = ((i*n2p+j)*n3p+k);
				
			if (myid ==0 && i==0)
				iglobal= n2p*(N1-2)+j; 
			else if (myid == nprocs - 1 && i==n1+1)
				iglobal= n2p*(1)+j; 
			else
				iglobal= n2p*(i-1)+j+(myid*((n1g-1)*n2p));	
				
			if (myid < (int)((double)nprocs/4.0))
			{
			if((myid==0 && i==0 && j==1) || (myid==0 && i==1 && j==0))
				{
				trojBcA(i,j,ind,iglobal,dg);
				}
			else if((myid==((int)((double)nprocs/4.0)-1) && i==n1-1 && j==n2) || (myid==((int)((double)nprocs/4.0)-1) && i==n1 && j==n2-1))
				{
				trojBcA(i,j,ind,iglobal,dg);
				}
			else
				trojBcB(i,j,ind,iglobal,dg);
			}
			else if ((myid >= (int)((double)nprocs/4.0)) && (myid < (int)((double)nprocs/2.0)) )
			{
			if(((myid == (int)((double)nprocs/4.0)) && i==0 && j==n2-1) || ((myid == (int)((double)nprocs/4.0)) && i==1 && j==n2))
				{
				trojBcB(i,j,ind,iglobal,dg);
				}
			else if((myid==((int)((double)nprocs/2.0)-1) && i==n1-1 && j==0) || (myid==((int)((double)nprocs/2.0)-1) && i==n1 && j==1))
				{
				trojBcB(i,j,ind,iglobal,dg);
				}
			else
				trojBcA(i,j,ind,iglobal,dg);
			}
			else if ((myid >= (int)((double)nprocs/2.0)) && (myid < (int)(3.0*(double)nprocs/4.0)) )
			{
			if(((myid == (int)((double)nprocs/2.0)) && i==0 && j==1) || ((myid == (int)((double)nprocs/2.0)) && i==1 && j==0))
				{
				trojBcA(i,j,ind,iglobal,dg);
				}
			else if((myid==((int)(3.0*(double)nprocs/4.0)-1) && i==n1-1 && j==n2) || (myid==((int)(3.0*(double)nprocs/4.0)-1) && i==n1 && j==n2-1))
				{
				trojBcA(i,j,ind,iglobal,dg);
				}
			else
				trojBcB(i,j,ind,iglobal,dg);
			}
			else 
			{
			if(((myid == (int)(3.0*(double)nprocs/4.0)) && i==0 && j==n2-1) || ((myid == (int)(3.0*(double)nprocs/4.0)) && i==1 && j==n2))
				{
				trojBcB(i,j,ind,iglobal,dg);
				}
			else if((myid==(nprocs-1) && i==n1-1 && j==0) || (myid==(nprocs-1) && i==n1 && j==1))
				{
				trojBcB(i,j,ind,iglobal,dg);
				}
			else
				trojBcA(i,j,ind,iglobal,dg);
			}		
			}
			
	free (dg);	

if(myid==0) 	printf(" OK\n");	
}

void associate_TU_BC()
{
	int64_t i, j, k, ind, iglobal, index;
	int64_t indu;

		
	for (j=0;j<=n2+1;j++)
		for (i=0;i<=n1+1;i++)
			{
			ind = ((i*n2p + j)*n3p + n3-1);

			b[ind] = 0.0; 
			x[ind] = 0.0;
					
			index = ((i*n2p + j)*n3p + n3-1)*36+13;	
				
			for(k=index-13;k<=index+22;k++)
				MC[k]=0.0;
	
			MC[index]=1.0;
		
			}	
}

void write_T(int c)
{
	int64_t i,j, ind, iglobal;
	double *dTD,*TDw;

	dTD = (double *)calloc((ceil((double)Ne1/ nprocs)) * n2, sizeof(double));

	
	if (myid == 0) TDw = (double *)calloc( nprocs* (ceil((double)Ne1/ nprocs)) * n2, sizeof(double));


if (myid == 0) printf(" Write ");		

	for (i = 2; i <= n1; i++)
		for (j = 1; j <= n2; j++)		
		{
			ind = ((i*n2p + j)*n3p + 0);
			iglobal= n2*(i-2)+j-1; 
			dTD[iglobal] = x[ind];
		}

	MPI_Gather(dTD,(ceil((double)Ne1/ nprocs)) * n2, MPI_DOUBLE, TDw, (ceil((double)Ne1/ nprocs)) * n2, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
	
if (myid == 0)
	{
		sprintf(name, "%d_T_%s_%dx%dx%d.dat",c,WRITE_prefix,Ne1,Ne2,Ne3);
		fw = fopen(name, "w");

		for (j = 0; j < n2; j++)
			for (i = 0; i < N1; i++)
				{
				
				if (i==0)
				{
					ind = ((n2p + j+1)*n3p + 0);				
					fprintf(fw,"%.18f\n",x[ind]);
				}
				else
				{
					ind = ((i-1)*n2+j);				
					fprintf(fw, "%.18f\n",TDw[ind]);
				}
			}
		fclose(fw);
		free(TDw);
		
	}
	free(dTD);
	
	if (myid == 0) 
	{	
		printf("T OK \n");
	}		
}

void write_TU(int c)
{
	int64_t i,j, ind, iglobal;
	double *dTD,*TDw;

	dTD = (double *)calloc((ceil((double)Ne1/ nprocs)) * n2, sizeof(double));

	
	if (myid == 0) TDw = (double *)calloc( nprocs* (ceil((double)Ne1/ nprocs)) * n2, sizeof(double));


if (myid == 0) printf(" Write ");		

	for (i = 2; i <= n1; i++)
		for (j = 1; j <= n2; j++)		
		{
			ind = ((i*n2p + j)*n3p + Nu);
			iglobal= n2*(i-2)+j-1; 
			dTD[iglobal] = x[ind];
		}

	MPI_Gather(dTD,(ceil((double)Ne1/ nprocs)) * n2, MPI_DOUBLE, TDw, (ceil((double)Ne1/ nprocs)) * n2, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
	
if (myid == 0)
	{
		sprintf(name, "%d_TU_%s_%dx%dx%d.dat",c,WRITE_prefix,Ne1,Ne2,Ne3);
		fw = fopen(name, "w");

		for (j = 0; j < n2; j++)
			for (i = 0; i < N1; i++)
				{
				
				if (i==0)
					{
					ind = ((n2p + j+1)*n3p + Nu);				
					fprintf(fw,"%.18f\n",x[ind]);
					}
				else
					{
					ind = ((i-1)*n2+j);				
					fprintf(fw, "%.18f\n",TDw[ind]);
					}
		}
		
		fclose(fw);
		free(TDw);
		
	}
	free(dTD);
	
	if (myid == 0) 
	{	
		printf("T OK \n");
	}		
}

int get_memory_usage_kb(long* vmrss_kb, long* vmsize_kb)
{
    /* source: https://hpcf.umbc.edu/general-productivity/checking-memory-usage/#heading_toc_j_3 */
	/* Get the the current process' status file from the proc filesystem */
    FILE* procfile = fopen("/proc/self/status", "r");

    long to_read = 8192;
    char buffer[to_read];
    int read = fread(buffer, sizeof(char), to_read, procfile);
    fclose(procfile);

    short found_vmrss = 0;
    short found_vmsize = 0;
    char* search_result;

    /* Look through proc status contents line by line */
    char delims[] = "\n";
    char* line = strtok(buffer, delims);

    while (line != NULL && (found_vmrss == 0 || found_vmsize == 0) )
    {
        search_result = strstr(line, "VmRSS:");
        if (search_result != NULL)
        {
            sscanf(line, "%*s %ld", vmrss_kb);
            found_vmrss = 1;
        }

        search_result = strstr(line, "VmSize:");
        if (search_result != NULL)
        {
            sscanf(line, "%*s %ld", vmsize_kb);
            found_vmsize = 1;
        }

        line = strtok(NULL, delims);
    }

    return (found_vmrss == 1 && found_vmsize == 1) ? 0 : 1;
}

int get_cluster_memory_usage_kb(long* vmrss_per_process, long* vmsize_per_process, int root, int np)
{
	   /* source: https://hpcf.umbc.edu/general-productivity/checking-memory-usage/#heading_toc_j_3 */
    long vmrss_kb;
    long vmsize_kb;
    int ret_code = get_memory_usage_kb(&vmrss_kb, &vmsize_kb);

    if (ret_code != 0)
    {
        printf("Could not gather memory usage!\n");
        return ret_code;
    }

    MPI_Gather(&vmrss_kb, 1, MPI_UNSIGNED_LONG, 
        vmrss_per_process, 1, MPI_UNSIGNED_LONG, 
        root, MPI_COMM_WORLD);

    MPI_Gather(&vmsize_kb, 1, MPI_UNSIGNED_LONG, 
        vmsize_per_process, 1, MPI_UNSIGNED_LONG, 
        root, MPI_COMM_WORLD);

    return 0;
}

int get_global_memory_usage_kb(long* global_vmrss, long* global_vmsize, int np)
{
    /* source: https://hpcf.umbc.edu/general-productivity/checking-memory-usage/#heading_toc_j_3 */
	long vmrss_per_process[np];
    long vmsize_per_process[np];
    int ret_code = get_cluster_memory_usage_kb(vmrss_per_process, vmsize_per_process, 0, np);

    if (ret_code != 0)
    {
        return ret_code;
    }

    *global_vmrss = 0;
    *global_vmsize = 0;
    for (int i = 0; i < np; i++)
    {
        *global_vmrss += vmrss_per_process[i];
        *global_vmsize += vmsize_per_process[i];
    }

    return 0;
}

void copy(double *x, double *y)
{
	int64_t i,j,k, ind;
	#pragma omp parallel for private(j,k,ind)	
	for (i=1;i<=n1;i++)
		for (j=1;j<=n2;j++)
			for (k=0;k<n3;k++)
			{
				ind = ((i*n2p+j)*n3p+k);
				y[ind] = x[ind];
            }
}

void x_xpay(double *x, double alfa, double *y)
{
	int64_t i,j,k,ind;
	#pragma omp parallel for private(j,k,ind)	
	for (i=1;i<=n1;i++)
		for (j=1;j<=n2;j++)
			for (k=0;k<n3;k++)
			{
				ind = ((i*n2p+j)*n3p+k);
				x[ind] += alfa * y[ind];
            }
}

void x_ymax(double *x, double alfa, double *y)
{
	int64_t i,j,k,ind;
	#pragma omp parallel for private(j,k,ind)	
	for (i=1;i<=n1;i++)
		for (j=1;j<=n2;j++)
			for (k=0;k<n3;k++)
			{
				ind = ((i*n2p+j)*n3p+k);
				x[ind] = y[ind] - alfa*x[ind];
            }
}

double dot(double *x, double *y)
{
	int64_t i,j,k,ind;
	double tmp=0.0;
	double sum=0.0;
	#pragma omp parallel for private(j,k,ind) reduction(+:tmp)
	for (i=1;i<=n1;i++)
		for (j=1;j<=n2;j++)
			for (k=0;k<n3;k++)
			{
				ind = ((i*n2p+j)*n3p+k);
				tmp+=x[ind]*y[ind];  
			}	
			
	MPI_Allreduce(&tmp,&sum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); 
	return sum;  
}

double norm2(double *x)
{
	int64_t i,j,k,ind;
	double tmp=0.0;
	double sum=0.0;
	#pragma omp parallel for private(j,k,ind) reduction(+:tmp)
	for (i=1;i<=n1;i++)
		for (j=1;j<=n2;j++)
			for (k=0;k<n3;k++)
			{
			ind = ((i*n2p+j)*n3p+k);
			tmp+=x[ind]*x[ind]; 
			
		}
	
	MPI_Allreduce(&tmp,&sum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); 

	return sqrt(sum);  
}

void y_Ax(double *x, double *y)
{
	int64_t i,j,k;
	int64_t ind, index; 

	#pragma omp parallel for private(j,k,ind,index) 
	for (i=1;i<=n1;i++)
		for (j=1;j<=n2;j++)
			for (k=0;k<n3;k++)
			{
			ind = ((i*n2p+j)*n3p+k);
			index = ((i*n2p + j)*n3p + k)*36+13;
			
				y[ind] = (
				x[ind-dwe-dsn-1] * MC[index-13] +
				x[ind-dwe-dsn] * MC[index-12] + 
				x[ind-dwe-dsn+1] * MC[index-11]+
				x[ind-dwe-1] * MC[index-10]+
				x[ind-dwe] * MC[index-9]+
				x[ind-dwe+1] * MC[index-8]+
				x[ind-dwe+dsn-1] * MC[index-7]+
				x[ind-dwe+dsn] * MC[index-6]+
				x[ind-dwe+dsn+1] * MC[index-5]+
				x[ind-dsn-1] * MC[index-4]+
				x[ind-dsn] * MC[index-3]+
				x[ind-dsn+1] * MC[index-2]+
				x[ind-1] * MC[index-1]+
				x[ind] * MC[index]+
				x[ind+1] * MC[index+1]+
				x[ind+dsn-1] * MC[index+2]+
				x[ind+dsn] * MC[index+3]+
				x[ind+dsn+1] * MC[index+4]+
				x[ind+dwe-dsn-1] * MC[index+5]+
				x[ind+dwe-dsn] * MC[index+6]+
				x[ind+dwe-dsn+1] * MC[index+7]+
				x[ind+dwe-1] * MC[index+8]+
				x[ind+dwe] * MC[index+9]+
				x[ind+dwe+1] * MC[index+10]+
				x[ind+dwe+dsn-1] * MC[index+11]+
				x[ind+dwe+dsn] * MC[index+12]+
				x[ind+dwe+dsn+1] * MC[index+13]+
				x[ind-dwe-dsn+2] * MC[index+14]+
				x[ind-dwe+2] * MC[index+15]+
				x[ind-dwe+dsn+2] * MC[index+16]+
				x[ind-dsn+2] * MC[index+17]+
				x[ind+2] * MC[index+18]+
				x[ind+dsn+2] * MC[index+19]+
				x[ind+dwe-dsn+2] * MC[index+20]+
				x[ind+dwe+2] * MC[index+21]+
				x[ind+dwe+dsn+2] * MC[index+22]
				);
				
			} 
}

void y_bAx(double *x, double *y)
{
	int64_t i,j,k,kk;
	int64_t ind, index; 

	#pragma omp parallel for private(j,k,kk,ind,index)
	for (i=1;i<=n1;i++)
		for (j=1;j<=n2;j++)
			for (k=0;k<n3;k++)
			{
			ind = ((i*n2p+j)*n3p+k);
			index = ((i*n2p + j)*n3p + k)*36+13;
			
		
				y[ind] = b[ind] - (
				x[ind-dwe-dsn-1] * MC[index-13] +
				x[ind-dwe-dsn] * MC[index-12] + 
				x[ind-dwe-dsn+1] * MC[index-11]+
				x[ind-dwe-1] * MC[index-10]+
				x[ind-dwe] * MC[index-9]+
				x[ind-dwe+1] * MC[index-8]+
				x[ind-dwe+dsn-1] * MC[index-7]+
				x[ind-dwe+dsn] * MC[index-6]+
				x[ind-dwe+dsn+1] * MC[index-5]+
				x[ind-dsn-1] * MC[index-4]+
				x[ind-dsn] * MC[index-3]+
				x[ind-dsn+1] * MC[index-2]+
				x[ind-1] * MC[index-1]+
				x[ind] * MC[index]+
				x[ind+1] * MC[index+1]+
				x[ind+dsn-1] * MC[index+2]+
				x[ind+dsn] * MC[index+3]+
				x[ind+dsn+1] * MC[index+4]+
				x[ind+dwe-dsn-1] * MC[index+5]+
				x[ind+dwe-dsn] * MC[index+6]+
				x[ind+dwe-dsn+1] * MC[index+7]+
				x[ind+dwe-1] * MC[index+8]+
				x[ind+dwe] * MC[index+9]+
				x[ind+dwe+1] * MC[index+10]+
				x[ind+dwe+dsn-1] * MC[index+11]+
				x[ind+dwe+dsn] * MC[index+12]+
				x[ind+dwe+dsn+1] * MC[index+13]+
				
				x[ind-dwe-dsn+2] * MC[index+14]+
				x[ind-dwe+2] * MC[index+15]+
				x[ind-dwe+dsn+2] * MC[index+16]+
				x[ind-dsn+2] * MC[index+17]+
				x[ind+2] * MC[index+18]+
				x[ind+dsn+2] * MC[index+19]+
				x[ind+dwe-dsn+2] * MC[index+20]+
				x[ind+dwe+2] * MC[index+21]+
				x[ind+dwe+dsn+2] * MC[index+22]
				);
			} 
}

void precondMatrix()
{
	int64_t i,j,k;
	int64_t ind, index; 

	#pragma omp parallel for private(j,k,ind,index)
	for (i=1;i<=n1;i++)
		for (j=1;j<=n2;j++)
			for (k=0;k<n3;k++)
			{
			ind = ((i*n2p+j)*n3p+k);
			index = ((i*n2p + j)*n3p + k)*36+13;
			
			b[ind]=b[ind]/MC[index];
			
			
			MC[index-13]= MC[index-13] /MC[index];
			MC[index-12]= MC[index-12]/MC[index] ; 
			MC[index-11]= MC[index-11]/MC[index];
			MC[index-10]= MC[index-10]/MC[index];
			MC[index-9]= MC[index-9]/MC[index];
			MC[index-8]= MC[index-8]/MC[index];
			MC[index-7]= MC[index-7]/MC[index];
			MC[index-6]= MC[index-6]/MC[index];
			MC[index-5]= MC[index-5]/MC[index];
			MC[index-4]= MC[index-4]/MC[index];
			MC[index-3]= MC[index-3]/MC[index];
			MC[index-2]= MC[index-2]/MC[index];
			MC[index-1]= MC[index-1]/MC[index];
			
			MC[index+1]=  MC[index+1]/MC[index];
			MC[index+2]=  MC[index+2]/MC[index];
			MC[index+3]=  MC[index+3]/MC[index];
			MC[index+4]=  MC[index+4]/MC[index];
			MC[index+5]=  MC[index+5]/MC[index];
			MC[index+6]=  MC[index+6]/MC[index];
			MC[index+7]=  MC[index+7]/MC[index];
			MC[index+8]=  MC[index+8]/MC[index];
			MC[index+9]=  MC[index+9]/MC[index];
			MC[index+10]=  MC[index+10]/MC[index];
			MC[index+11]=  MC[index+11]/MC[index];
			MC[index+12]= MC[index+12]/MC[index];
			MC[index+13]= MC[index+13]/MC[index];
			
			MC[index+14]=  MC[index+14]/MC[index];
			MC[index+15]=  MC[index+15]/MC[index];
			MC[index+16]=  MC[index+16]/MC[index];
			MC[index+17]=  MC[index+17]/MC[index];
			MC[index+18]=  MC[index+18]/MC[index];
			MC[index+19]=  MC[index+19]/MC[index];
			MC[index+20]= MC[index+20]/MC[index];
			MC[index+21]= MC[index+21]/MC[index];
			MC[index+22]= MC[index+22]/MC[index];
		
			
			MC[index]=MC[index]/MC[index];
			} 
}

void BICGstabL(int L)
{
	/* Developed using: https://etna.math.kent.edu/vol.1.1993/pp11-32.dir/pp11-32.pdf */
	
	if (myid == 0) printf("BICGstabL solver: \n\n");
  
	int64_t i,j,k,ij,ind;
	
	double *gamma = (double *) malloc( (L+1) * sizeof(double));
    double *gamma_p = (double *) malloc( (L+1) * sizeof(double));
    double *gamma_pp = (double *) malloc( (L+1) * sizeof(double));
    double *tau = (double *) malloc( (L*L) * sizeof(double));
    double *sigma = (double *) malloc( (L+1) * sizeof(double));
	double *rtilde = (double *) malloc(n1p * n2p * n3p * sizeof(double));				
	double **r = (double **) malloc((L+1) * sizeof(double *));
	double **u = (double **) malloc((L+1) * sizeof(double *));

	for(i = 0; i <= L; i++)
	{
  		gamma[i]=0.0;
  		gamma_p[i]=0.0;
		gamma_pp[i]=0.0;
		sigma[i]=0.0;
  	}
	
	for(i = 0; i < L*L; i++)
		tau[i]=0.0;
		
	for(i = 0; i <= L; i++)
		{
		r[i] = (double *) malloc(n1p * n2p * n3p * sizeof(double));
		u[i] = (double *) malloc(n1p * n2p * n3p * sizeof(double));
		}

	#pragma omp parallel for private(j,k,ind)	
	for (i=0;i<n1p;i++)
		for (j=0;j<n2p;j++)
			for (k=0;k<n3p;k++)
			{
			ind = ((i*n2p + j)*n3p + k);
			rtilde[ind]=0.0;
			}

	for(ij = 0; ij <= L; ij++)
	{
		#pragma omp parallel for private(j,k,ind)	
		for (i=0;i<n1p;i++)
			for (j=0;j<n2p;j++)
				for (k=0;k<n3p;k++)
				{
				ind = ((i*n2p + j)*n3p + k);
				r[ij][ind]=0.0;
				u[ij][ind]=0.0;
				}
	}
	
	double resid,rho1,beta;
	double rho = 1.0, alpha = 0.0, omega = 1.0;
	double breaktol = 1e-30;
	iter=0;
	
	y_bAx(x,r[0]);
	copy(r[0],rtilde);

	get_global_memory_usage_kb(&global_vmrss, &global_vmsize, nprocs-1);
  	if (myid == 0)
		printf("\nGlobal memory usage: VmRSS = %6ld KB, VmSize = %6ld KB\n\n", global_vmrss, global_vmsize);
	
	while (((resid = norm2(r[0])) > tol )&&(iter<max_it)) 
	{  
  
    if ((iter % pertol == 0)&&(myid == 0))
		printf("%d\t %.15lf \n",iter,resid);

	if (iter % pertol_write == 0)
		write_T(iter);

    iter+=1;
	
    rho = -omega * rho;
    for (j = 0; j < L; j++) 
	{
		if (fabs(rho) < breaktol) 
		{ 
			printf("small rho %d %.10lf\n",iter,rho); 
			goto finish; 
		}

		rho1 = dot(r[j],rtilde);
		
		beta = alpha * rho1 / rho;
		rho = rho1;

		for (i = 0; i <= j; ++i)
				x_ymax(u[i],beta,r[i]);
			
		send(u[j]);
		y_Ax(u[j], u[j+1]);
		alpha = rho / dot(u[j+1], rtilde);
		
       for (i = 0; i <= j; ++i)
			x_xpay(r[i], -alpha, u[i+1]);
		
		send(r[j]);
        y_Ax(r[j], r[j+1]);
        x_xpay(x, alpha, u[0]);   
    }
	
    for (j = 1; j <= L; ++j) 
	{
		for (i = 1; i < j; ++i) 
		{
			ij = (j-1)*L + (i-1);
			tau[ij] = dot(r[j], r[i]) / sigma[i];
			x_xpay(r[j], -tau[ij], r[i]);
        }
        sigma[j] = dot(r[j],r[j]);
        gamma_p[j] = dot(r[0], r[j]) / sigma[j];
    }
	
    omega = gamma[L] = gamma_p[L];
		
    for (j = L-1; j >= 1; j--) 
		{
        gamma[j] = gamma_p[j];
        for (i = j+1; i <= L; i++)
			gamma[j] -= tau[(i-1)*L + (j-1)] * gamma[i];
        }
	
    for (j = 1; j < L; ++j) 
		{
        gamma_pp[j] = gamma[j+1];
        for (i = j+1; i < L; i++)
			gamma_pp[j] += tau[(i-1)*L + (j-1)] * gamma[i+1];
		}
		
    x_xpay(x, gamma[1], r[0]);
    x_xpay(r[0], -gamma_p[L], r[L]);
    x_xpay(u[0], -gamma[L], u[L]);
	
    for (j = 1; j < L; ++j) 
	{ 
      x_xpay(x, gamma_pp[j], r[j]);
      x_xpay(r[0], -gamma_p[j], r[j]);
      x_xpay(u[0], -gamma[j], u[j]);
    }
    if (iter == max_it ) goto finish;

 }
    finish:;
	if(myid == 0)
	{
		printf("Solved : %d\t %.15lf \n",iter,resid);
		printf(" ---------------------------------------------\n");
	}
	
	free(rtilde);
	free(sigma);
	free(tau);
	for(i = 0; i <= L; i++)
		{
			free(u[i]);
			free(r[i]);
		}
	free(u);
	free(r);
	free(gamma);
	free(gamma_p);
	free(gamma_pp);
}

void set_up()
{
	int64_t to, i, j, k, ind, ii, index;
		
	ne1 = (ceil((double)Ne1/ nprocs));	/*number of nodes per procesor*/
	ne2 = Ne2;							
	ne3 = Ne3;
	N1 = Ne1 + 1;
	n1 = ne1 + 1;
	n1g = ne1 + 1;						
	n2 = ne2 + 1;
	N2 = ne2 + 1;
	n3 = ne3 + 1;
	n1p = ne1 + 3;						
	n2p = ne2 + 3;
	n3p = ne3 + 1;
	
	if (myid == 0)
		printf("Nproc: %d \t n1:%d\t n2:%d\t n3:%d \n", nprocs,N1,n2,n3);

	dwe = n2p*n3p;
	dsn = n3p;
	
	to = n1;
	if (myid == nprocs - 1)
		{
			to = Ne1 - (nprocs - 1)*(n1-1) +1;
			n1 = to;
		}
	
	if ((myid == 0) || (myid == nprocs - 1))
		printf("Myid: %d \t n1:%d\t n2:%d\t n3:%d \n", myid,n1,n2,n3);
	

	node = (struct Nodes*) malloc(n1p *n2p * n3p * sizeof(struct Nodes));
	x = (double *)malloc(n1p * n2p * n3p * sizeof(double));
	b = (double *)malloc(n1p * n2p * n3p * sizeof(double));
		
	#pragma omp parallel for private(j,k,ind)
	for (i=0;i<n1p;i++)
		for (j=0;j<n2p;j++)
			for (k=0;k<n3p;k++)
			{
			ind = ((i*n2p+j)*n3p+k);
			
			x[ind]=0.0;
			b[ind]=0.0;

			}
	
	get_global_memory_usage_kb(&global_vmrss, &global_vmsize, nprocs-1);
	if (myid == 0)
    {
        printf("\nGlobal memory usage: VmRSS = %6ld KB, VmSize = %6ld KB\n\n", global_vmrss, global_vmsize);
    }
	MPI_Barrier(MPI_COMM_WORLD);
	
	MC = (double *)malloc(37 * n1p * n2p * n3p * sizeof(double)); /* non-zero coefficients of Matrix */
	
	#pragma omp parallel for
	for(ii=0;ii<=36 * n1p * n2p * n3p ;ii++)
		MC[ii]=0.0;
	
	#pragma omp parallel for private(j,k,ind)
	for (i=0;i<=n1+1;i++)
		for (j=0;j<=n2+1;j++)
			for (k=0;k<=n3;k++)
			{
				index = ((i*n2p + j)*n3p + k)*36+13;
				MC[index]=0.0;
			}
	}

int main(int argc,char *argv[])
{
	int64_t i, j, k, ind, ii, index;
	
	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	
	if(numa_available()!=-1)
	{
		numa_run_on_node(myid%numa_num_configured_cpus()%numa_num_configured_nodes());
		numa_set_preferred(myid%numa_num_configured_cpus()%numa_num_configured_nodes());
	}
	
	prev = myid - 1;
	next = myid + 1;
	if (myid == nprocs - 1)
		next = 0;
	if (myid == 0)
		prev = nprocs - 1;
	
	name = (char *)malloc(100);
	
	start = MPI_Wtime();
	
	set_up();

	
	if (myid == 0)
	{
		printf(" ---------------------------------------------\n");
		printf("Initialization OK \n");
	}
			
	XYZnodes();
	
	get_global_memory_usage_kb(&global_vmrss, &global_vmsize, nprocs-1);
	if (myid == 0)
		printf("\nGlobal memory usage: VmRSS = %6ld KB, VmSize = %6ld KB\n\n", global_vmrss, global_vmsize);
	
	elements();
	
	if (myid == 0) printf("\n");

	associate_dg_BC();	
	associate_TU_BC();
	
	free(node);	
	
	system("emptyram");
	
	get_global_memory_usage_kb(&global_vmrss, &global_vmsize, nprocs-1);
	if (myid == 0)
		printf("\nGlobal memory usage: VmRSS = %6ld KB, VmSize = %6ld KB\n\n", global_vmrss, global_vmsize);
	
	if (myid == 0) 
	{
		printf(" ---------------------------------------------\n");
		printf("Solving with ");
	}
	
	precondMatrix();
	BICGstabL(8); /* used BICGstabL(l) with l=8*/
		
	write_T(iter);
	write_TU(iter);
			
	end=MPI_Wtime();
	
	MPI_Barrier(MPI_COMM_WORLD);

	printf("Processor %d, time: %e secs \n",myid,end-start);
	
	free(x);
	free(b);
	free(B);
	free(L);
	free(H);
	free(MC);

	MPI_Finalize();
}
