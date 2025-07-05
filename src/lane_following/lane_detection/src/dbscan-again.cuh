#include <cwchar>
#include <algorithm>
#include <iostream>
#include <vector>
#include <exception>
#include <cuda_runtime.h>
#include <queue>
#include <fstream>
#include <unistd.h>
#define SIZE 64
#define MAX_NODES 1000000
typedef unsigned char uchar;

__device__ __managed__ int true_count = 1;

#define __START_TIMER__                                                                            \
    float gpu_elapsed_time_ms;                                                                     \
    cudaEvent_t start, stop;                                                                       \
    cudaEventCreate(&start);                                                                       \
    cudaEventCreate(&stop);                                                                        \
    cudaEventRecord(start, 0);

#define __END_TIMER__                                                                              \
    cudaEventRecord(stop, 0);                                                                      \
    cudaEventSynchronize(stop);                                                                    \
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);                                       \
    std::cout << "GPU Time: " << gpu_elapsed_time_ms << " ms\n";

template<typename T>
__global__ void num_neighbors(int *count_list, T *points, int *adj_list, int offset, int no_of_nodes, float eps){
    int index = blockIdx.x*SIZE + threadIdx.x;
    int blk_ind = threadIdx.x;
    int glob_stride = offset*index;
    float x,y;
    int temp=0;
    T dist;
    if(index < no_of_nodes){
        x=points[index<<1];
        y=points[(index<<1) + 1];
    }
    __shared__ T l1pts[SIZE*2];
    for(int blk_off=0; blk_off < no_of_nodes;blk_off+=SIZE){
        int tmp = blk_ind + blk_off;
        __syncthreads();
        if(tmp < no_of_nodes){
            l1pts[blk_ind<<1] = points[(tmp)<<1];
            l1pts[(blk_ind<<1) + 1] = points[(tmp<<1) + 1];
        }
        __syncthreads();
        int size = min((no_of_nodes - blk_off), SIZE);
        if(index < no_of_nodes){
            for(int i=0;i<size && (temp < offset);++i){
                dist = fabs(x-l1pts[i<<1]) + fabs(y-l1pts[(i<<1) + 1]);
                if(dist <= eps){
                    adj_list[glob_stride + temp] = i + blk_off;
                    temp++;
                }
            }
        }
    }
    if(index < no_of_nodes){
        count_list[index]=temp;
    }
}

template<typename T>
class DBSCAN{
private:
	float eps;
	int min_pts;
	int no_nodes;
	int MAX_NEIGHBORS;
	int OFFSET;
	double dmin;
	int *labels;
	uchar *v;
	T *dev_nodes;
	int *adj_list;
	int *dev_adj_list;
	int *dev_neighbor_count;
	std::vector<int> neighbor_count;
public:
    DBSCAN(float eps, int min_pts) : eps(eps), min_pts(min_pts) {
        std::cout << "[DBSCAN] Initializing...\n";
        this->dmin = 0.301; // 0.847
        this->OFFSET = 1.868 * eps * eps / (dmin * dmin); //4
        this->MAX_NEIGHBORS = (MAX_NODES * OFFSET) + 1;

        std::cout << "[DBSCAN] Parameters - eps: " << eps << ", min_pts: " << min_pts 
                  << ", OFFSET: " << OFFSET << ", MAX_NEIGHBORS: " << MAX_NEIGHBORS << "\n";

        // if(eps > 1){
        //     std::cerr << "[WARNING] Large eps value: " << eps << "\n";
        //     this->dmin = 0.05;
        // }

        cudaMalloc(&dev_nodes, 2 * sizeof(T)*MAX_NODES);
        std::cout << "[DBSCAN] Device memory allocated for nodes\n";

        cudaMalloc(&dev_neighbor_count, sizeof(int)*MAX_NODES);
        std::cout << "[DBSCAN] Device memory allocated for neighbor count\n";

        adj_list = new int[MAX_NEIGHBORS];
        cudaMalloc(&dev_adj_list, sizeof(int)*(MAX_NEIGHBORS));
        std::cout << "[DBSCAN] Device memory allocated for adjacency list\n";

        labels = new int[MAX_NODES];
        neighbor_count = std::vector<int>(MAX_NODES, 0);
    }

    ~DBSCAN() {
        cudaFree(dev_adj_list);
        cudaFree(dev_nodes);
        cudaFree(dev_neighbor_count);
        delete []labels;
        delete []adj_list;
        std::cout << "[DBSCAN] Cleaned up memory\n";
    }

    int identify_cluster(std::vector<T> &nodes, int &out_clusters) {
        std::cout << "[DBSCAN] Starting clustering...\n";
        no_nodes = std::min(MAX_NODES, static_cast<int>(nodes.size()) / 2);
        std::cout << "[DBSCAN] Using " << no_nodes << " nodes\n";

        cudaMemcpy(dev_nodes, nodes.data(), 2 * sizeof(T)*no_nodes, cudaMemcpyHostToDevice);
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess){
            std::cerr << "[CUDA ERROR] memcpy dev_nodes: " << cudaGetErrorString(err) << std::endl;
        }
        std::cout<<"After err check for memcpy dev_nodes\n";
        __START_TIMER__
        dim3 dim_block(SIZE, 1, 1);
        dim3 dim_grid((no_nodes + SIZE-1)/SIZE, 1, 1);

        std::cout<<"BEFORE num_neighbours\n";
        num_neighbors<T><<<dim_grid, dim_block>>>(dev_neighbor_count, dev_nodes, dev_adj_list, OFFSET, no_nodes, eps);
        std::cout<<"AFTER num_neighbours\n";
        cudaDeviceSynchronize();
        __END_TIMER__
        std::cout<<"AFTER TIMER\n";

        err = cudaGetLastError();
        if(err != cudaSuccess){
            std::cerr << "[CUDA ERROR] Kernel execution: " << cudaGetErrorString(err) << std::endl;
        }

        cudaMemcpy(neighbor_count.data(), dev_neighbor_count, sizeof(int)*no_nodes, cudaMemcpyDeviceToHost);
        cudaMemcpy(adj_list, dev_adj_list, sizeof(int) * MAX_NEIGHBORS, cudaMemcpyDeviceToHost);

        std::cout << "[DBSCAN] Neighbor data copied from device to host\n";

        for(int i=0;i < no_nodes; ++i) labels[i] = 0;

        int cluster_id = 1;
        std::queue<int> q;
        std::cout << "[DBSCAN] Beginning BFS for clusters...\n";

        for(int node=0; node < no_nodes; ++node){
            if(!labels[node] && neighbor_count[node] >= min_pts){
                q.push(node);
                labels[node] = cluster_id;

                while(!q.empty()){
                    int curr_node = q.front(); q.pop();
                    for(int i = curr_node * OFFSET; i < curr_node * OFFSET + neighbor_count[curr_node]; ++i){
                        int neighbor = adj_list[i];
                        if(!labels[neighbor]){
                            labels[neighbor] = cluster_id;
                            if(neighbor_count[neighbor] >= min_pts) q.push(neighbor);
                        }
                    }
                }
                std::cout << "[DBSCAN] Formed cluster " << cluster_id << "\n";
                cluster_id++;
            }
        }

        std::cout << "[DBSCAN] Assigning noise points to separate clusters\n";
        for(int node=0; node < no_nodes; ++node){
            if(labels[node] == 0) labels[node] = cluster_id++;
        }

        std::cout << "[DBSCAN] Clustering completed. Total clusters: " << cluster_id - 1 << "\n";
        out_clusters = cluster_id - 1;
        return no_nodes;

    }

    void show_labels() {
        std::cout << "Labels:\n";
        for(int i = 0; i < no_nodes; ++i){
            std::cout << (int)labels[i] << ' ';
        }
        std::cout << '\n';
    }

    uchar label(int index){
        if(index >= MAX_NODES){
            std::cerr << "[DBSCAN] WARNING: Index out of range in label(): " << index << "\n";
            return no_nodes + rand()%no_nodes;
        }
        return labels[index];
    }
};
