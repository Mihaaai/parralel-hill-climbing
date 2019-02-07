/*EXPECTED FULL OUTPUT :
Example with comments explaining lines

{HILL CLIMBING LINE} = world_rank x y
{HILL CLIMBING SECTION} = ({HILL CLIMBING LINE}\n)*

world_size
{HILL CLIMBING SECTION} // sequential algorithm
BS best_score x y  //best score of sequential algorithm
FS duration // final of sequential algorithm, followed by time duration
{HILL CLIMBING SECTION} // parrallel algorithm
BP best_score x y  //best score of parrallel algorithm
FP duration // final of parralel algorithm, followed by time duration
*/

// #include "pch.h"
#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>
#include <unistd.h>
// #include <windows.h> 

#define DEBUG_OUTPUT false
#define DEBUG_EPOCHS false
#define DEBUG_CANDIDATES false
#define FULL_OUTPUT true
#define M_PI 3.1415

using namespace std;

double func(double point[2]) {
	double phi = point[0];
	double p = point[1];
	double result = 35000 * sin(3 * phi) * sin(2 * p)
		+ 9700 * cos(10 * phi) * cos(20 * p)
		- 800 * sin(25 * phi + 0.03 * M_PI)
		+ 550 * cos(p + 0.2 * M_PI);
	return result;
}

double candidate(int c, int acceleration) {
	switch (c) {
	case 1:
		return -acceleration;
	case 2:
		return -1.0 / acceleration;
	case 3:
		return 0;
	case 4:
		return 1 / acceleration;
	case 5:
		return acceleration;
	default:
		break;
	}
}

void printStatus(double currentPoint[2], double stepSize[2]) {
	printf("Current Point : (%.2f, %.2f). Step size : (%.2f, %.2f). \n",
		currentPoint[0], currentPoint[1], stepSize[0], stepSize[1]);
}

double* hillClimb(double *finalScore, int coordX, int coordY, bool isParralel) {
	sleep(50);
	// Sleep(50);
	// number of iterations with no moves before we conclude that we converged
	int burnoutEpochs = 100;
	double stepSize[2] = { 0.5, 0.5 };
	double acceleration = 2;	// same acceleration for both dimensions ???
	double epsilon = 0.1;

	// define initial starting point
	double *currentPoint = new (nothrow) double[2];
	if (!currentPoint) {
		return NULL;
	}
	currentPoint[0] = coordX;
	currentPoint[1] = coordY;

	int epoch = 1;
	int iterations = 0;
	double lastImprovement = 0;

	//get rank of current process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	if(FULL_OUTPUT && isParralel){
		printf("%d %.2f %.2f\n", world_rank, currentPoint[0], currentPoint[1]);	
	}
	while (true) {
		// compute initial score
		double before = func(currentPoint);
		for (int i = 0; i < 2; ++i) {
			int best = -1; //best candidate
			double bestScore = -numeric_limits<double>::infinity();  // minus INF
			double currentValue = currentPoint[i];
			for (int j = 1; j <= 5; ++j) {
				currentPoint[i] = currentPoint[i] + stepSize[i] * candidate(j, acceleration);
				double temp = func(currentPoint);
				if (DEBUG_CANDIDATES) {
					printf("Dimension %d - candidate %d\n", i, j);
					printStatus(currentPoint, stepSize);
					printf("Value : %.2f\n", temp);
					printf("\n");
				}
				currentPoint[i] = currentValue;	//revert changes
				if (temp > bestScore) {
					bestScore = temp;
					best = j;
				}

			}
			if (DEBUG_CANDIDATES) {
				printf("Dimension %d - best candidate %d\n", i, best);
			}
			if (candidate(best, acceleration) == 0) {
				stepSize[i] = stepSize[i] / acceleration;
			}
			else {
				currentPoint[i] = currentPoint[i] + stepSize[i] * candidate(best, acceleration);
				stepSize[i] = stepSize[i] * candidate(best, acceleration); // accelerate
				if(FULL_OUTPUT && isParralel){
					printf("%d %.2f %.2f\n", world_rank, currentPoint[0], currentPoint[1]);	
				}	
			}

		}

		double now = func(currentPoint);
		double improvement = now - before;
		++iterations;
		if (DEBUG_EPOCHS) {
			printf("Epoch %d !!!!!!!!!!!!!!!!!!!!\n", epoch);
			printf("Now : %.2f - improvement : %.2f \n", now, improvement);
			printStatus(currentPoint, stepSize);
		}
		if (improvement < epsilon) {
			if (epoch < burnoutEpochs) {
				++epoch;
			}
			else {
				*finalScore = now;
				return currentPoint;
			}
		}
		else {
			epoch = 0;
			lastImprovement = improvement;
		}

	}
}

void sequential_climbing(int count) {
	if(DEBUG_OUTPUT){
		printf("\nSequential climbing:\n");	
	}
	double score;
	double bestClimber[2], best_score = 0;
	for (int i = 0; i < count; i++) {
		double *climber = hillClimb(&score, rand() % 100, rand() % 100, false);
		if (score > best_score) {
			best_score = score;
			bestClimber[0] = climber[0];
			bestClimber[1] = climber[1];
		}
	}
	if(DEBUG_OUTPUT){
		printf("Finest climber at %.2f meters : (%.2f, %.2f)\n", best_score, bestClimber[0], bestClimber[1]);	
	}
	if(FULL_OUTPUT){
		printf("BS %.2f %.2f %.2f\n",best_score, bestClimber[0], bestClimber[1]);
	}
}

void parallel_climbing(int world_rank, int world_size) {

	if (world_rank != 0) {
		double score;
		double *climber = hillClimb(&score, rand() % 100, rand() % 100, true);

		MPI_Send(&score, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		MPI_Send(&climber[0], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		MPI_Send(&climber[1], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
	else {
		double best_phi, best_p, best_score = 0;
		for (int process = 1; process < world_size; process++) {
			double phi, p, score;
			MPI_Recv(&score, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);
			MPI_Recv(&phi, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);
			MPI_Recv(&p, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);

			if (best_score < score) {
				best_score = score;
				best_phi = phi;
				best_p = p;
			}
		}

		if(DEBUG_OUTPUT){
			printf("Finest climber at %.2f meters : (%.2f, %.2f)\n", best_score, best_phi, best_p);	
		}
		if(FULL_OUTPUT){
			printf("BP %.2f %.2f %.2f\n",best_score, best_phi, best_p);
		}
	}
}


int main(int argc, char** argv) {
	// Initialize the MPI environment
	MPI_Init(NULL, NULL);
	// Find out rank, size
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	srand(time(NULL) + world_rank);

	if (world_rank == 0) {
		printf("%d\n",world_size);
		double sequential_time_start, sequential_time_end;
		if (argc == 2) {
			sequential_time_start = MPI_Wtime();
			sequential_climbing(atoi(argv[1]));
			sequential_time_end = MPI_Wtime();
			if(DEBUG_OUTPUT){
				printf("Execution time for sequential climbing: %.4f seconds\n", sequential_time_end - sequential_time_start);	
			}
			if(FULL_OUTPUT){
				printf("FS %.4f\n", sequential_time_end - sequential_time_start);  // Sequential algo duration
			}
			
		}
		if(DEBUG_OUTPUT){
			printf("\nParallel climbing:\n");	
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	double parallel_time_start, parallel_time_end;
	if (world_rank == 0) {
		parallel_time_start = MPI_Wtime();
	}
	parallel_climbing(world_rank, world_size);

	if (world_rank == 0) {
		parallel_time_end = MPI_Wtime();
		
		if(DEBUG_OUTPUT){
			printf("Execution time for parallel climbing: %.4f seconds\n", parallel_time_end - parallel_time_start);	
		}
		if(FULL_OUTPUT){
			printf("FP %.4f\n", parallel_time_end - parallel_time_start); // Parralel algo duration
		}
	}
	MPI_Finalize();
}
