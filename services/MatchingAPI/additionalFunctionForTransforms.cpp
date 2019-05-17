#include "additionalFunctionForTransforms.h"

void poly2_addPoint(double *Matr, double *B, double x, double y, double xi, double yi, bool is_bilin)
{
    double sec_coef[NUM_POLY2_PARAMS];
    int i, j, K;
    K = NUM_POLY2_PARAMS / 2;
    sec_coef[0] = 1.0;
    sec_coef[1] = x;
    sec_coef[2] = y;
    sec_coef[4] = x * y;
    if(!is_bilin) 
    {
        sec_coef[3] = x * x;
        sec_coef[5] = y * y;
    }
    else
    {
        sec_coef[3] = 0.0;
        sec_coef[5] = 0.0;
    }
    for(i = 0; i < K; i++) 
    {
        for(j = 0; j < K; j++) 
        {
            Matr[i * NUM_POLY2_PARAMS + j] += sec_coef[j] * sec_coef[i];
            Matr[(i + K) * NUM_POLY2_PARAMS + (j + K)] += sec_coef[j] * sec_coef[i];
        }
        B[i] += xi * sec_coef[i];
        B[i + K] += yi * sec_coef[i];
    }
    if(is_bilin) 
    {
        Matr[3 * NUM_POLY2_PARAMS + 3] = Matr[5 * NUM_POLY2_PARAMS + 5] =
        Matr[9 * NUM_POLY2_PARAMS + 9] = Matr[11 * NUM_POLY2_PARAMS + 11] = 1.0;
    }
}

bool SolveLinear(double *matrix, double *b, vector<double> * sol, int s)
{
    int i, j, k, l, best_j;			
    double max, tmp, mult, *mt1, *mt2;		
    for(j = 0; j < s - 1; j++) 
    {
        if(fabs(matrix[j * s + j]) < ZERO_VALUE) 
        {
            for(max = 0.0, best_j = -1, l = j; l < s; l++) 
            {
                if(fabs(matrix[l * s + j]) > max) 
                {
                    max = fabs(matrix[l * s + j]);
                    best_j = l;
                }
            }
            if(best_j == -1 || fabs(max) < ZERO_VALUE) { return ER_DETERM_ZERO; }
            if(best_j != j) 
            {
                for(k = j; k < s; k++) 
                {
                    tmp = matrix[j * s + k];
                    matrix[j * s + k] = matrix[best_j * s + k];
                    matrix[best_j * s + k] = tmp;
                }
                tmp = b[j]; b[j] = b[best_j]; b[best_j] = tmp;
            }
        }
        for(l = j + 1; l < s; l++) 
        {
            mt1 = matrix + l * s + j;
            mt2 = matrix + j * s + j;
            mult = (*mt1) / (*mt2);
            for(k = j; k < s; k++, mt1++, mt2++) { (*mt1) -= (*mt2) * mult; }
            b[l] -= b[j] * mult;
        }
    }
    if(fabs(matrix[(s - 1) * s + (s - 1)]) < ZERO_VALUE) { return ER_DETERM_ZERO; }
    b[s - 1] /= matrix[(s - 1) * s + (s - 1)];
    for(j = s - 2; j >= 0; j--) 
    {
        mt1 = matrix + j * s + (j + 1);
        for(i = j + 1; i < s; i++, mt1++) { b[j] -= (*mt1) * b[i]; }
        b[j] /= matrix[j * s + j];
    }
    for (i = 0; i < s; i++)
    {
        (*sol).push_back(b[i]);
    }
    return SUCCESSFUL;
}


void zeroEquations(double *Matr, double *B, int s)
{
    int i, j;
    for(i = 0; i < s; i++) {
        for(j = 0; j < s; j++) { Matr[i * s + j] = 0.0; }
        B[i] = 0.0;
    }
}

double calcArrayAver(vector<double> x)
{
    int N = x.size(), i;
    double Sx;
    for(i = 0, Sx = 0.0; i < N; i++) 
    {
        Sx += x[i];
    }
    return Sx / N;
}

double calcArraysMultAver(vector<double> x, vector<double> y)
{
    int N = x.size(), i;		
    double XY;	
    for(i = 0, XY = 0; i < N; i++) {
        XY += x[i] * y[i];
    }
    return XY / N;
}

double calcArraysHomoAver(vector<double> x1, vector<double> y1, vector<double> x2, vector<double> y2)
{
    return calcArraysMultAver(x1, x2) + calcArraysMultAver(y1, y2);
}

void calcFourArraysAver(vector<double> x1, vector<double> y1, vector<double> x2, vector<double> y2, double *Ax1, double *Ay1, double *Ax2, double *Ay2)
{
    *Ax1 = calcArrayAver(x1);
    *Ay1 = calcArrayAver(y1);
    *Ax2 = calcArrayAver(x2);
    *Ay2 = calcArrayAver(y2);
}

double calcArraysHeteroAver(vector<double> x1, vector<double> y1, vector<double> x2, vector<double> y2)
{
    return (calcArraysMultAver(x1, y2) - calcArraysMultAver(x2, y1));
}

void getShiftByRotScale(double Ax1, double Ay1, double Ax2, double Ay2, double scale, double angle, double *xsh, double *ysh)
{
    double Acos, Asin;
    Acos = scale * cos(angle);
    Asin = scale * sin(angle);
    *xsh = Ax2 - Ax1 * Acos - Ay1 * Asin;
    *ysh = Ay2 + Ax1 * Asin - Ay1 * Acos;
}

double similar_sumErr(vector<double> x1, vector<double> y1, vector<double> x2, vector<double> y2, double scale, double angle, double xsh, double ysh)
{			
    double sum, Acos, Asin, dx, dy;		
    int N = x1.size(), i;
    Acos = scale * cos(angle);
    Asin = scale * sin(angle);
    for(i = 0, sum = 0.0; i < N; i++) 
    {
        dx = x2[i] - Acos * x1[i] - Asin * y1[i] - xsh;
        dy = y2[i] + Asin * x1[i] - Acos * y1[i] - ysh;
        sum += SQR_SUM(dx, dy);
    }
    return sum;
}
