/*********************************************************************************
Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
All rights reserved.
  
Permission to use, copy, modify and distribute this software and its documentation for 
educational purpose is hereby granted without fee, provided that the above copyright 
notice and this permission notice appear in all copies of this software and that you do 
not sell the software.
  
THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
OTHERWISE.

The CUDA Kernel for Applying BFS on a loaded Graph. Created By Pawan Harish
**********************************************************************************/
#ifndef _KERNEL2_H_
#define _KERNEL2_H_

__global__ void
Kernel2( bool* g_graph_mask, bool *g_updating_graph_mask, bool* g_graph_visited, bool *g_over, int no_of_nodes)
{
	int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	
	printf("Kernel#2 - Thread Id = %d - Before Check\n", tid);
	printf("Kernel#2 - Thread Id = %d - Number of Nodes (no_of_nodes) = %d\n", tid, no_of_nodes);
	printf("Kernel#2 - Thread Id = %d - Update Graph Mask (g_updating_graph_mask[tid]) = %d\n", tid, g_updating_graph_mask[tid]);

	if( tid<no_of_nodes && g_updating_graph_mask[tid])
	{
		g_graph_mask[tid]=true;
		g_graph_visited[tid]=true;
		*g_over=true;
		g_updating_graph_mask[tid]=false;

		printf("Kernel#2 - Thread Id = %d - After Check\n", tid);
		printf("Kernel#2 - Thread Id = %d - Update Graph Mask (g_updating_graph_mask[tid]) = %d\n", tid, g_updating_graph_mask[tid]);
		printf("Kernel#2 - Thread Id = %d - Visited Flag (g_graph_visited[tid]) = %d\n", tid, g_graph_visited[tid]);
		printf("Kernel#2 - Thread Id = %d - Over Flag (g_over) = %d\n", tid, g_over);
		printf("Kernel#2 - Thread Id = %d - Graph Mask (g_graph_mask[tid]) = %d\n", tid, g_graph_mask[tid]);
	}
}

#endif

