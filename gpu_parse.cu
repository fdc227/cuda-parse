#include "utilities.cuh"


#define num 31
#define totalwidth 500
#define xwidth 50
#define elenum 11
#define timesteps 5001

using namespace std;

void vectorcpy(double* a, std::vector<double>& vec, size_t size)
{
    for(size_t i = 0; i < size; i ++)
    {
        a[i] = vec[i];
    }
}

__global__ void gpu_compute(double* y_d, double* y_dot_d, double* y_torsion_d, double* X, double* y_gen, double* torsion_gen) // 500 threads per block, per thread calculates one value for y and torsion
{
    int M = threadIdx.x / xwidth;
    int N = threadIdx.x % xwidth;
    int blockidx = blockDim.x * blockIdx.x;
    int blockidy = blockDim.y * blockIdx.y; // representing time
    int index_x = blockidx + threadIdx.x;
    int index_y = blockidy + threadIdx.y;

    int total_index = index_x + totalwidth * index_y;

    __shared__ double y_local[elenum];
    __shared__ double y_dot_local[elenum];
    __shared__ double y_torsion_local[elenum];
    __shared__ double X_local[xwidth];

    if (threadIdx.x < elenum * 3 + xwidth)
    {
        if (threadIdx.x < elenum)
        {
            y_local[threadIdx.x] = y_d[elenum * index_y + threadIdx.x];
            if (blockIdx.x == 0)
            {
                printf("ylocal[%d]=%f", threadIdx.x, y_d[elenum * index_y + threadIdx.x]);
                /*cout << "y_local[" << threadIdx.x << "] = " << y_d[elenum * index_y + threadIdx.x] << "copied into cache" << endl;;*/
            }
        }
        else if (threadIdx.x < elenum * 2)
        {
            y_dot_local[threadIdx.x - elenum] = y_dot_d[elenum * index_y + threadIdx.x - elenum];
 /*           if (blockIdx.x == 0)
            {
                cout << "y_dot_local[" << threadIdx.x - elenum << "] = " << y_dot_d[elenum * index_y + threadIdx.x - elenum] << "copied into cache" << endl;;
            }*/
        }
        else if (threadIdx.x < elenum * 3)
        {
            y_torsion_local[threadIdx.x - elenum * 2] = y_torsion_d[elenum * index_y + threadIdx.x - 2 * elenum];
            //if (blockIdx.x == 0)
            //{
            //    cout << "y_torsion_local[" << threadIdx.x - elenum*2 << "] = " << y_torsion_d[elenum * index_y + threadIdx.x - 2 * elenum] << "copied into cache" << endl;;
            //}
        }
        else
        {
            X_local[threadIdx.x - elenum * 3] = X[threadIdx.x - elenum * 3];
      /*      if (blockIdx.x == 0)
            {
                cout << "X_local[" << threadIdx.x - elenum * 3 << "] = " << X[threadIdx.x - elenum * 3] << "copied into cache" << endl;;
            }*/
        }
    }
    __syncthreads();

    double q0 = y_local[M];
    double q0_dot = y_dot_local[M];
    double q0t = y_torsion_local[M];
    double q1 = y_local[M + 1];
    double q1_dot = y_dot_local[M + 1];
    double q1t = y_torsion_local[M + 1];
    double x = X_local[N];

    double y, torsion;

    y = q0 * (1 - 3 * pow(x, 2) + 2 * pow(x, 3)) + q0_dot * (x - 2 * pow(x, 2) + pow(x, 3)) + q1 * (3 * pow(x, 2) - 2 * pow(x, 3)) + q1_dot * (-pow(x, 2) + pow(x, 3));
    torsion = q0t * (1 - x) + q1t * x;

    y_gen[total_index] = y;
    torsion_gen[total_index] = torsion;
}

__global__ void gpu_compute_simple(double* y_d, double* y_dot_d, double* y_torsion_d, double* X, double* y_gen, double* torsion_gen)
{
    int array_idx = elenum * blockIdx.y + threadIdx.x;

    double q0 = y_d[array_idx];
    double q1 = y_d[array_idx + 1];
    double q0_dot = y_dot_d[array_idx];
    double q1_dot = y_dot_d[array_idx + 1];
    double qt0 = y_torsion_d[array_idx];
    double qt1 = y_torsion_d[array_idx + 1];

    int out_idx = xwidth *(elenum - 1) * blockIdx.y + xwidth * threadIdx.x;
    for (int i = 0; i < xwidth; i++)
    {
        double x = X[i];
        double y, torsion;
        y = q0 * (1 - 3 * pow(x, 2) + 2 * pow(x, 3)) + q0_dot * (x - 2 * pow(x, 2) + pow(x, 3)) + q1 * (3 * pow(x, 2) - 2 * pow(x, 3)) + q1_dot * (-pow(x, 2) + pow(x, 3));
        torsion = qt0 * (1 - x) + qt1 * x;
        y_gen[out_idx + i] = y;
        torsion_gen[out_idx + i] = torsion;
    }
}

int main(void)
{
    string filename = "sol.dat";
    ifstream file;
    file.open(filename, ios::in);
    vector<double> t;
    vector<double> y;
    vector<double> y_dot;
    vector<double> y_torsion;
    double local;

DATA:
    {
        for (int i = 0; i < num; i++)
        {
            if (!file.eof())
            {
                file >> local;
                if (i == 0)
                {
                    t.push_back(local);
                    y.push_back(0.0);
                    y_dot.push_back(0.0);
                    y_torsion.push_back(0.0);
                }
                else if ((i - 1) % 3 == 0)
                {
                    y.push_back(local);
                }
                else if ((i - 1) % 3 == 1)
                {
                    y_dot.push_back(local);
                }
                else
                {
                    y_torsion.push_back(local);
                }

                if (i == num - 1)
                    goto DATA;
            }
            else
                goto EXIT;
        }
    }
EXIT:

 /*   cout << t.size() << endl;
    cout << y.size() << endl;
    cout << y_dot.size() << endl;
    cout << y_torsion.size() << endl;

    cout << t.back() << endl;
    cout << y.back() << endl;
    cout << y_dot.back() << endl;
    cout << y_torsion.back() << endl;*/

    size_t yn = y.size();
    size_t tn = y_torsion.size();

    // Host vectors
    double* t_h, * y_h, * y_dot_h, * y_torsion_h, * X_h, * y_gen_h, * y_torsion_gen_h;
    t_h = new double[t.size()];
    y_h = new double[y.size()];
    y_dot_h = new double[y_dot.size()];
    y_torsion_h = new double[y_torsion.size()];
    X_h = new double[xwidth];
    y_gen_h = new double[(elenum - 1) * xwidth * timesteps];
    y_torsion_gen_h = new double[(elenum - 1) * xwidth * timesteps];


    vectorcpy(t_h, t, t.size());
    vectorcpy(y_h, y, y.size());
    vectorcpy(y_dot_h, y_dot, y_dot.size());
    vectorcpy(y_torsion_h, y_torsion, y_torsion.size());

    write_array_to_file(y_h, "y_h.dat", 5001, 11);
    write_array_to_file(y_dot_h, "y_dot_h.dat", 5001, 11);
    write_array_to_file(y_torsion_h, "y_torsion_h.dat", 5001, 11);

    cout << y_h[0] << endl;
    cout << y_dot_h[0] << endl;
    cout << y_torsion_h[0] << endl;

    cout << y_h[yn-1] << endl;
    cout << y_dot_h[yn-1] << endl;
    cout << y_torsion_h[yn-1] << endl;



    for (int i = 0; i < xwidth; i++)
    {
        X_h[i] = 1.0 / double(xwidth) * double(i);
    }

    void* host_array[6] = { (void*)y_h, (void*)y_dot_h, (void*)y_torsion_h, (void*)X_h, (void*)y_gen_h, (void*)y_torsion_gen_h };
    void* host_array2[4] = {(void*)y_h, (void*)y_dot_h, (void*)y_torsion_h, (void*)X_h };


    double *y_d, *y_dot_d, *y_torsion_d, *X_d, *y_gen_d, *y_torsion_gen_d;
    void** device_array[6] = {(void**)&y_d, (void**)&y_dot_d, (void**)&y_torsion_d, (void**)&X_d, (void**)&y_gen_d, (void**)&y_torsion_gen_d};
    void** device_array2[4] = { (void**)&y_d, (void**)&y_dot_d, (void**)&y_torsion_d, (void**)&X_d };
    size_t size[6] = { yn * sizeof(double), yn * sizeof(double), yn * sizeof(double), xwidth * sizeof(double), (elenum - 1) * xwidth * timesteps * sizeof(double), (elenum - 1) * xwidth * timesteps * sizeof(double) };
    size_t size2[4] = { yn * sizeof(double), yn * sizeof(double), yn * sizeof(double), xwidth * sizeof(double) };

    cudaError_t cudastatus;
    cudastatus = arrayMalloc(device_array, 6, size);
    cudastatus = arraycpyHtoD_v2(device_array2, host_array2, 4, size2);

    dim3 DimGrid(1, timesteps, 1);
    dim3 DimBlock((elenum-1), 1, 1);
    gpu_compute_simple <<<DimGrid, DimBlock >>> (y_d, y_dot_d, y_torsion_d, X_d, y_gen_d, y_torsion_gen_d) ;

    cudastatus = onecpyDtoH(y_gen_h, y_gen_d, (elenum - 1) * xwidth * timesteps * sizeof(double));
    cudastatus = onecpyDtoH(y_torsion_gen_h, y_torsion_gen_d, (elenum - 1) * xwidth * timesteps * sizeof(double));

    if (cudastatus != cudaSuccess)
    {
        fstream errfile;
        errfile.open("gpu_log.txt", ios::out);
        errfile << cudaGetErrorString(cudastatus) << endl;
    }

    /*while (cin.get() != 'q')
    {
        int n = cin.get();
        cout << "y[" << n << "] is " << y_gen_h[n] << endl;
        cout << "torsion[" << n << "] is " << y_torsion_gen_h[n] << endl;
    }*/

    //vector<double> Y(y_gen_h, y_gen_h + (elenum - 1) * xwidth * timesteps);
    //vector<double> T(y_torsion_gen_h, y_torsion_gen_h + (elenum - 1) * xwidth * timesteps);

    write_array_to_file(y_gen_h, "y.dat", timesteps, (elenum - 1)* xwidth);
    write_array_to_file(y_torsion_gen_h, "torsion.dat", timesteps, (elenum - 1)* xwidth);

    return 0;
}
