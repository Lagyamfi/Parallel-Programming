// BASELINE VERSION
//
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#ifndef FP64
#define FP32
#endif

//#ifdef FP64
typedef double real;
#define POW(x,y)  pow((x),(y))
#define SQRT(x)   sqrt((x))
//#else
//typedef float real;
//#define POW(x,y) powf((x),(y))
//#define SQRT(x)  sqrtf((x))
//#endif

//typedef double real;
#define SOFTENING_SQUARED  0.01

typedef struct { real x, y, z; }    real3;
typedef struct { real x, y, z, w; } real4;

real3 bodyBodyInteraction(real4 iPos, real4 jPos)
{
  real rx, ry, rz, distSqr, s;

  rx = jPos.x - iPos.x;  ry = jPos.y - iPos.y;  rz = jPos.z - iPos.z;

  distSqr = rx*rx+ry*ry+rz*rz;
//nmodification
  if (distSqr < SOFTENING_SQUARED) s = SQRT(SOFTENING_SQUARED);
  else                             s = SQRT(distSqr);

  s = s*s*s;
  s = jPos.w / s;
// end of modification
  real3 f;
  f.x = rx * s;  f.y = ry * s; f.z = rz * s;

  return f;
}


void integrate(real4 * out, real4 * in,
               real3 * vel, real3 * force,
               real    dt,  int n)
{
  int i, j;
  for (i = 0; i < n; i++)
  {
    real fx=0, fy=0, fz=0;

    for (j = 0; j < n; j++)
      //if (i!=j)
      {
        real3 ff = bodyBodyInteraction(in[i], in[j]);
        fx += ff.x;  fy += ff.y; fz += ff.z;
      }

    force[i].x = fx;  force[i].y = fy;  force[i].z = fz;
  }

  for (i = 0; i < n; i++)
  {
    real fx = force[i].x, fy = force[i].y, fz = force[i].z;
    real px = in[i].x,    py = in[i].y,    pz = in[i].z,    invMass = in[i].w;
    real vx = vel[i].x,   vy = vel[i].y,   vz = vel[i].z;

    // acceleration = force / mass;
    // new velocity = old velocity + acceleration * deltaTime
    vx += (fx * invMass) * dt;
    vy += (fy * invMass) * dt;
    vz += (fz * invMass) * dt;

    // new position = old position + velocity * deltaTime
    px += vx * dt;
    py += vy * dt;
    pz += vz * dt;

    out[i].x = px;
    out[i].y = py;
    out[i].z = pz;
    out[i].w = invMass;

    vel[i].x = vx;
    vel[i].y = vy;
    vel[i].z = vz;
  }
}


real dot(real v0[3], real v1[3])
{
  return v0[0]*v1[0]+v0[1]*v1[1]+v0[2]*v1[2];
}


real normalize(real vector[3])
{
  float dist = sqrt(dot(vector, vector));
  if (dist > 1e-6)
  {
    vector[0] /= dist;
    vector[1] /= dist;
    vector[2] /= dist;
  }
  return dist;
}


void cross(real out[3], real v0[3], real v1[3])
{
  out[0] = v0[1]*v1[2]-v0[2]*v1[1];
  out[1] = v0[2]*v1[0]-v0[0]*v1[2];
  out[2] = v0[0]*v1[1]-v0[1]*v1[0];
}


void randomizeBodies(real4* pos,
                     real3* vel,
                     float clusterScale,
                     float velocityScale,
                     int   n)
{
  srand(42);
  float scale = clusterScale;
  float vscale = scale * velocityScale;
  float inner = 2.5f * scale;
  float outer = 4.0f * scale;

  int p = 0, v=0;
  int i = 0;
  while (i < n)
  {
    real x, y, z;
    x = rand() / (float) RAND_MAX * 2 - 1;
    y = rand() / (float) RAND_MAX * 2 - 1;
    z = rand() / (float) RAND_MAX * 2 - 1;

    real point[3] = {x, y, z};
    real len = normalize(point);
    if (len > 1)
      continue;

    pos[i].x= point[0] * (inner + (outer - inner)*rand() / (real) RAND_MAX);
    pos[i].y= point[1] * (inner + (outer - inner)*rand() / (real) RAND_MAX);
    pos[i].z= point[2] * (inner + (outer - inner)*rand() / (real) RAND_MAX);
    pos[i].w= 1.0f;

    x = 0.0f;
    y = 0.0f;
    z = 1.0f;
    real axis[3] = {x, y, z};
    normalize(axis);

    if (1 - dot(point, axis) < 1e-6)
    {
      axis[0] = point[1];
      axis[1] = point[0];
      normalize(axis);
    }
    real vv[3] = {(real)pos[i].x, (real)pos[i].y, (real)pos[i].z};
    real vv0[3];

    cross(vv0, vv, axis);
    vel[i].x = vv0[0] * vscale;
    vel[i].y = vv0[1] * vscale;
    vel[i].z = vv0[2] * vscale;

    i++;
  }
}


real3 average(real4 * p, int n)
{
  int i;
  real3 av= {0.0, 0.0, 0.0};
  for (i = 0; i < n; i++)
  {
    av.x += p[i].x;
    av.y += p[i].y;
    av.z += p[i].z;
  }
  av.x /= n;
  av.y /= n;
  av.z /= n;
  return av;
}


int main(int argc, char** argv)
{
  int i, n = 20000;
  int iterations = 10;
  real dt = 0.01667;

  if (argc >= 2) n = atoi(argv[1]);
  if (argc >= 3) iterations = atoi(argv[2]);

  real4 *pin  = (real4*)malloc(n * sizeof(real4));
  real4 *pout = (real4*)malloc(n * sizeof(real4));
  real3 *v    = (real3*)malloc(n * sizeof(real3));
  real3 *f    = (real3*)malloc(n * sizeof(real3));

  randomizeBodies(pin, v,  1.54f, 8.0f, n);

  printf("n=%d bodies for %d iterations:\n", n, iterations);

  for (i = 0; i < iterations; i++)
  {
    integrate (pout, pin, v, f, dt, n);
    real4 *t = pout;
    pout = pin;
    pin = t;
  }

  real3 p_av= average( pin, n);
  printf("Average position: (%f,%f,%f)\n", p_av.x, p_av.y, p_av.z);
  printf("Body-0  position: (%f,%f,%f)\n", pin[0].x, pin[0].y, pin[0].z);

  free(pin);  free(pout);  free(v);  free(f);

  return 0;
}
