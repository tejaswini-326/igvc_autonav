/*
 *@Author: Krutarth Patel                                           
 *@Date: 20th December 2024
 *@Description : GDBSCAN header only library
 * 				using CUDA for speeeeeeeeeeeeeeed.
 */

#include <cwchar>
#include <algorithm>
#include <iostream>
#include <vector>
#include <exception>
#include <cuda_runtime.h>
#define SIZE 96
typedef unsigned char uchar;

__device__ __managed__ int true_count = 1;
#define __START_TIMER__                                                                            \
    float gpu_elapsed_time_ms;                                                                     \
    cudaEvent_t start, stop;                                                                       \
    cudaEventCreate(&start);                                                                       \
    cudaEventCreate(&stop);                                                                        \
    cudaEventRecord(start, 0);

#define __END_TIMERR_                                                                              \
    cudaEventRecord(stop, 0);                                                                      \
    cudaEventSynchronize(stop);                                                                    \
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);                                       \
    std::cout << "gpu time : " << gpu_elapsed_time_ms << "ms\n";

__global__ void search(int *adj_list, int *count, uchar *frontier, uchar *v, int max_neigh, float eps, int min_pts, int no_of_nodes){
    int index = blockIdx.x*SIZE + threadIdx.x;
    if(index==0)true_count=0;
    if(index < no_of_nodes){
        if(frontier[index]){  //if node is a frontier
            frontier[index] = 0; 
            int start = index * max_neigh;
            for(int neighbor=start;neighbor < start + count[index]; ++neighbor){ //set all its neighbors as frontiers
                if(!v[adj_list[neighbor]]){
                    //if border point
                    if(count[adj_list[neighbor]] >= min_pts){
                        frontier[adj_list[neighbor]] = 1;
                        true_count = 1;
                    }
                    v[adj_list[neighbor]] = 1;
                }
            }
        }
    }
}


__global__ void reset(int node, uchar *frontier, uchar *v){
    /*
    * kernel code to do bfs
    */
    int index = blockIdx.x*SIZE + threadIdx.x;
    int val=0;
    if(index == node)
        val=1;
    frontier[index] = val;
    v[index] = val;

    true_count = 1;
}

template<typename T>
__global__ void num_neighbors(int *count_list, T *points, int *adj_list, int offset, int no_of_nodes, float eps){
    int index = blockIdx.x*SIZE + threadIdx.x;
    if(index < no_of_nodes){
        int temp=0;
        int glob_stride = offset*index;
		T dist;
        float x=points[index<<1], y=points[(index<<1) + 1];
		// if a point has more neighbors than max we cut
		// I think this is valid since in my testing I haven't found
		// a case where we actually have a lot of neighbors
		// the pointcloud is pretty uniformly spaced
        for(int i=index+1;i < no_of_nodes && temp < offset; ++i){
            dist = abs(x-points[i<<1]);
            if(dist <= eps){
                dist+=abs(y-points[(i<<1) + 1]);
                if(dist <= eps){
                    adj_list[glob_stride + temp] = i;
                    temp++;
                }
            }else break;
        }
        for(int i=index-1;i > -1 && temp < offset; --i){
            dist = abs(x-points[i<<1]);
            if(dist <= eps){
                dist+=abs(y-points[(i<<1) + 1]);
                if(dist <= eps){
                    adj_list[glob_stride + temp] = i;
                    temp++;
                }
            }else break;
        }
        count_list[index]=temp;
    }
}

template<typename T>
class DBSCAN{
    private:
        float eps;
        int min_pts;
		int no_nodes;
		int MAX_NODES;     // maximum allowed nodes
		int MAX_NEIGHBORS; // thereotical limit on maximum total neighbor
		int OFFSET;        // max neighbor for one node
		double dmin;       // min distance between two nodes
        uchar *labels;
		uchar *frontier; 
		uchar *v; 
		T *dev_nodes;
        int *adj_list; 
		int *dev_neighbor_list;
    public:
        DBSCAN(float eps, int min_pts) :  eps(eps), min_pts(min_pts)
        {
			this->MAX_NODES = 30720;
			this->dmin = 0.008;
			this->MAX_NEIGHBORS = (MAX_NODES * (4 * eps * eps)/(dmin * dmin));
			this->OFFSET = 4 * eps * eps/(dmin * dmin);
			if(eps > 1){
				std::cerr << "WARNING: Are you sure you want to keep this big of an eps? " << eps << std::endl;
				std::cerr << "Yours truly, DBSCAN" << std::endl;
				// now, your scale might be different, so we will adjust to
				// that, the expectation is that eps/dmin is about 100-500
				// more may result in too much memory being allocated on the GPU
				this->dmin = 0.05;
			}
			
            //allocate nodes on device
            cudaMalloc(&dev_nodes, 2 * sizeof(T)*MAX_NODES);
            cudaMalloc(&dev_neighbor_list, sizeof(int)*MAX_NODES);
            // allocating memory to adjacency list
			// dmin is the minimum distance between two points,
			// eps is the distance considered for two points to be neighboring
			// The maximum neighbors a point could thus have is roughly 2 * eps/dmin
			// I measured dmin in rviz, and this is very rough, but 0.005 should be enough
			// trust...
            cudaMalloc(&adj_list, sizeof(int)*(MAX_NEIGHBORS));
            // allocating memory
            cudaMalloc(&frontier, sizeof(uchar)*MAX_NODES);
            cudaMalloc(&v, sizeof(uchar)*MAX_NODES);
        }
        
        ~DBSCAN()
        {
            cudaFree(adj_list);
            cudaFree(dev_nodes);
            cudaFree(dev_neighbor_list);
            cudaFree(frontier);
            cudaFree(v);
            delete []labels;
        }

        int identify_cluster(std::vector<T> &nodes){
			// this will be the effective size, We ignore all points 
			// after MAX_NODES!
			
#if defined(NDEBUG)
			if(nodes.size()/2 > MAX_NODES){
    			std::cerr << "Ignoring " << nodes.size() - MAX_NODES << " points!" << std::endl;
    			std::cerr << "Yours Truly, DBSCAN" << std::endl;
			}
#endif

			no_nodes = min(MAX_NODES,(int)nodes.size()/2);
			
            std::vector<int> neighbor_list(no_nodes, 0);
            //allocate nodes on device
            cudaMemcpy(dev_nodes, nodes.data(), 2 * sizeof(T)*no_nodes, cudaMemcpyHostToDevice);
			cudaError_t err = cudaGetLastError();
            //find neighbors
            dim3 dim_block(SIZE, 1, 1);
            dim3 dim_grid((no_nodes + SIZE-1)/SIZE, 1, 1);
            num_neighbors<T><<<dim_grid, dim_block>>>(dev_neighbor_list, dev_nodes, adj_list, this->OFFSET, no_nodes, eps);
            cudaMemcpy(neighbor_list.data(), dev_neighbor_list, sizeof(int)*no_nodes, cudaMemcpyDeviceToHost);
            
#if defined(NDEBUG)
            // allocating memory to adjacency list
            // this is for testing, I am not sure 
            // about the theoretical limit so I will
            // keep this for now. 
			std::cout << "no of nodes: " << no_nodes << std::endl; 
			for(int i=0;i < no_nodes; ++i){
				 std::cout << neighbor_list[i] << ' ';
			}
			std::cout << '\n';
		
            long long acc = 0;
            for(int i=0;i < no_nodes; ++i){
				acc += (long long)neighbor_list[i];
            }
			// NOTE: this will never happen since I changed
			// num_neighbors()
			if(acc >= MAX_NEIGHBORS){
				std::cerr << "Well, this is unexpected. There are more neighbors than theoretically possible!" << std::endl;
				std::cerr << "Theoretical limit: " << MAX_NEIGHBORS << ", total: " << acc << std::endl;
				std::cerr << "Two options, tune DBSCAN to match your reality OR tune eps and min_pts to match DBSCAN's reality" << std::endl;
				std::cerr << "Yours Truly, DBSCAN" << std::endl;
				return 0;
			}

			 //debug
			 //std::cout << "adj list :\n";
			 int *adj = new int[MAX_NEIGHBORS];
			 auto result = cudaMemcpy(adj, adj_list, sizeof(int)*MAX_NEIGHBORS, cudaMemcpyDeviceToHost);
			 std::cout << cudaGetErrorString(result) << std::endl;
			 for(int i=0;i < no_nodes; ++i){
			 for(int j=i * OFFSET; j < i * OFFSET + neighbor_list[i]; ++j){
				 std::cout << adj[j] << ' ';
			 }
			 std::cout << '\n';
			 }
			 delete []adj;
		
#endif

            this->labels = new uchar[no_nodes];
            int cluster_id = 1;
            for(int i=0;i < no_nodes; ++i){
                labels[i]=0;
            }        
            int neighbors;
            for(int node=0;node < no_nodes; ++node){
                neighbors = neighbor_list[node];
                if(!labels[node] && neighbors >= min_pts){
                    labels[node] = cluster_id; 
                    bfs(node, cluster_id++);
                }
            }
			// we return the number of nodes
			// considered in the algorithm
			return no_nodes;
        }
        
        void show_labels(){
            std::cout << "labels :\n";
            for(int i=0;i < no_nodes; ++i){
                std::cout << (int)labels[i] << ' ';
            }
            std::cout << '\n';
        }

        uchar label(int index){
			if(index >= MAX_NODES){
				// we need to think what to do
				// here, because it could be the case
				// that we have too many points and we do 
				// not consider some of them, I could
				// give a random value each time to make
				// sure no FALSE POSITIVES are considered 
				// at least
				return no_nodes + rand()%no_nodes;
			}
            return labels[index];
        }
        
	private:
        
        void bfs(int node, int cluster_id)
        {
            /*
            * start from a node and do bfs
            */
            dim3 dim_block(SIZE, 1);
            dim3 dim_grid((no_nodes + SIZE-1)/SIZE);
            reset<<<dim_grid,dim_block>>>(node, frontier, v);
			cudaDeviceSynchronize();
            while(true_count){
                search<<<dim_grid, dim_block>>>(adj_list,dev_neighbor_list,frontier,v,this->OFFSET,eps,min_pts,no_nodes);
                cudaDeviceSynchronize();
            }
            //back to host
            uchar V[no_nodes];
            cudaMemcpy(V, v, sizeof(uchar)*no_nodes, cudaMemcpyDeviceToHost);        
        
            for(int node=0;node < no_nodes; ++node){
                if(V[node]){
                    labels[node] = cluster_id;
                }
            }
        }
};
