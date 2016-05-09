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
#ifndef _KERNEL_H_
#define _KERNEL_H_

__global__ void
Kernel( Node* g_graph_nodes, int* g_graph_edges, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, int no_of_nodes) 
{
	int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	
	// printf("Kernel#1 - Thread Id = %d - Before Check and Loop\n", tid);
	// printf("Kernel#1 - Thread Id = %d - Graph Mask (g_graph_mask[tid]) = %d\n", tid, g_graph_mask[tid]);
	// printf("Kernel#1 - Thread Id = %d - Number of Nodes (no_of_nodes) = %d\n", tid, no_of_nodes);
	
	if( tid<no_of_nodes && g_graph_mask[tid])
	{
		g_graph_mask[tid]=false;

		// printf("Kernel#1 - Thread Id = %d - After Check\n", tid);
		// printf("Kernel#1 - Thread Id = %d - Start (g_graph_nodes[tid].starting) = %d\n", tid, g_graph_nodes[tid].starting);
		// printf("Kernel#1 - Thread Id = %d - Count (g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting) = %d\n",
		//  tid, (g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting));
		// printf("Kernel#1 - Thread Id = %d - Number of Edges (g_graph_nodes[tid].no_of_edges) = %d\n",
		//  tid, (g_graph_nodes[tid].no_of_edges));
		
		for(int i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++)
		{
			int id = g_graph_edges[i];

			// printf("Kernel#1 - Thread Id = %d - Enter Loop\n", tid);
			// printf("Kernel#1 - Thread Id = %d - Edge Id (g_graph_edges[i]) = %d\n", tid, g_graph_edges[i]);
			// printf("Kernel#1 - Thread Id = %d - Visited Flag (g_graph_visited[id]) = %d\n", tid, g_graph_visited[id]);

			if(!g_graph_visited[id])
			{
				g_cost[id]=g_cost[tid]+1;
				g_updating_graph_mask[id]=true;

				// printf("Kernel#1 - Thread Id = %d - Cost (g_cost[id]) = %d\n", tid, g_cost[id]);
			}
		}
	}
}

#endif 
