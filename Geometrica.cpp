#define RUN_ON_WINDOWS false
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

#if RUN_ON_WINDOWS
	#include "pch.h"
	#include <windows.h>
#else 
	#include <unistd.h>
#endif

#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <iostream>

#define MAX_ITERATIONS 1000
#define DEBUG_EPOCHS false
#define DEBUG_CANDIDATES false
#define DEBUG_LAST_IMPROV false
#define DEBUG_ITERATIONS false
#define M_PI 3.1415
#define DEBUG_OUTPUT false
#define FULL_OUTPUT true
#define FULL_OUTPUT_2 true

using namespace std;

struct Point
{
	double phi, p;
};

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

double* hillClimb(double *finalScore, int coordX, int coordY, bool doPrint) {
	
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

	if(FULL_OUTPUT && doPrint){
		printf("%d %.2f %.2f\n", world_rank, currentPoint[0], currentPoint[1]);	
	}

	while (true) {
		// compute initial score
		double before = func(currentPoint);
		for (int i = 0; i < 2; ++i) {
			int best = -1; //best candidate
			double bestScore = - numeric_limits<double>::infinity();  // minus INF
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
				if(FULL_OUTPUT && doPrint){
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
		if( iterations < MAX_ITERATIONS){
			return currentPoint;
		}

	}
}

//Sequential implementation for hill climbing
void sequential_climbing(Point points[], int nr_points) {

	if(DEBUG_OUTPUT){
		printf("\nSequential climbing:\n");	
	}
	
	double score;
	double bestClimber[2], best_score = 0;
	for (int i = 0; i < nr_points; i++) {
		double *climber = hillClimb(&score, points[i].phi, points[i].p, false);
		if (score > best_score) {
			best_score = score;
			bestClimber[0] = climber[0];
			bestClimber[1] = climber[1];
		}
	}
	if(FULL_OUTPUT){
		printf("BS %.2f %.2f %.2f\n",best_score, bestClimber[0], bestClimber[1]);
	}
	if(DEBUG_OUTPUT){
		printf("Finest climber at %.2f meters : (%.2f, %.2f)\n", best_score, bestClimber[0], bestClimber[1]);	
	}
}

//Parallel implementation for hill climbing
void parallel_climbing(int world_rank, int world_size, Point points[], int nr_points) {
	//Slave
	if (world_rank != 0) {
		while (true) {
			double phi, p, scope;
			MPI_Recv(&scope, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);
			if (scope != 1) {
				break;
			}
			MPI_Recv(&phi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);
			MPI_Recv(&p, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);

			double score;
			double *climber = hillClimb(&score, phi, p, true);

			MPI_Send(&score, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			MPI_Send(&climber[0], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			MPI_Send(&climber[1], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		}
		
	}
	//Master
	else {
		int contor = 0, process = 1;
		double best_phi = -1, best_p = -1, best_score = 0;
		//If there are more points than processes
		if (nr_points >= world_size) {
			//Where points are stil available
			while (contor < nr_points) {
				//Send first points to processes
				if (contor < world_size - 1) {
					double scope = 1;
					MPI_Send(&scope, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
					MPI_Send(&points[contor].phi, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
					MPI_Send(&points[contor].p, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
					contor++;
					process++;
				}
				//Receive message from process and send another point
				else {
					if (process % world_size == 0) {
						process++;
					}
					double phi, p, score;
					MPI_Recv(&score, 1, MPI_DOUBLE, process % world_size, 0, MPI_COMM_WORLD,
						MPI_STATUS_IGNORE);
					MPI_Recv(&phi, 1, MPI_DOUBLE, process % world_size, 0, MPI_COMM_WORLD,
						MPI_STATUS_IGNORE);
					MPI_Recv(&p, 1, MPI_DOUBLE, process % world_size, 0, MPI_COMM_WORLD,
						MPI_STATUS_IGNORE);

					if (best_score < score) {
						best_score = score;
						best_phi = phi;
						best_p = p;
					}

					double scope = 1;
					MPI_Send(&scope, 1, MPI_DOUBLE, process % world_size, 0, MPI_COMM_WORLD);
					MPI_Send(&points[contor].phi, 1, MPI_DOUBLE, process % world_size, 0, MPI_COMM_WORLD);
					MPI_Send(&points[contor].p, 1, MPI_DOUBLE, process % world_size, 0, MPI_COMM_WORLD);

					process++;
					contor++;
				}
			}

			//Compute the messages from the other processes in order
			for (int i = 1; i < world_size; i++) {
				if ((process + i) % world_size == 0) {
					process++;
				}
				double phi, p, score;
				MPI_Recv(&score, 1, MPI_DOUBLE, (process + i) % world_size, 0, MPI_COMM_WORLD,
					MPI_STATUS_IGNORE);
				MPI_Recv(&phi, 1, MPI_DOUBLE, (process + i) % world_size, 0, MPI_COMM_WORLD,
					MPI_STATUS_IGNORE);
				MPI_Recv(&p, 1, MPI_DOUBLE, (process + i) % world_size, 0, MPI_COMM_WORLD,
					MPI_STATUS_IGNORE);

				if (best_score < score) {
					best_score = score;
					best_phi = phi;
					best_p = p;
				}

				//Stop the other process
				double scope = 0;
				MPI_Send(&scope, 1, MPI_DOUBLE, (process + i) % world_size, 0, MPI_COMM_WORLD);
			}
		}
		//If there are more processes than points
		else {
			for (int process = 1; process < world_size; process++) {
				//Send the points to first processes
				if (contor < nr_points) {
					double scope = 1;
					MPI_Send(&scope, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
					MPI_Send(&points[contor].phi, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
					MPI_Send(&points[contor].p, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
					contor++;
				}
				//Stop the other processs
				else {
					double scope = 0;
					MPI_Send(&scope, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
					contor++;
				}
			}
			//Receive messages from processes in order
			for (int process = 1; process < nr_points + 1; process++) {
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

				double scope = 0;
				MPI_Send(&scope, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
			}
		}
		if(DEBUG_OUTPUT){
			printf("Finest climber at %.2f meters : (%.2f, %.2f)\n", best_score, best_phi, best_p);			
		}

		if(FULL_OUTPUT_2){
			printf("BEP %.2f %.2f %.2f\n",best_score, best_phi, best_p);
		}
	}
}

//Perfect parallel implementation for hill climbing
void perfect_parallel_climbing(int world_rank, int world_size, Point points[], int nr_points) {
	//Slave
	if (world_rank != 0) {
		while (true) {
			double phi, p, scope;
			MPI_Recv(&scope, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);
			if (scope != 1) {
				break;
			}
			MPI_Recv(&phi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);
			MPI_Recv(&p, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);

			double score;
			double *climber = hillClimb(&score, phi, p, false);

			MPI_Send(&score, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			MPI_Send(&climber[0], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			MPI_Send(&climber[1], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		}
	}
	//Master
	else {
		int contor = 0, process = 1;
		double best_phi = -1, best_p = -1, best_score = 0;

		int rank = 0, *processes;
		processes = new (nothrow) int[world_size];
		for (int i = 1; i < world_size; i++) {
			processes[i] = 0;
		}
		bool is_score = false, is_phi = false;

		//If there are more points than processes
		if (nr_points >= world_size) {
			//Send first points to their processes
			for (int process = 1; process < world_size; process++) {
				double scope = 1;
				MPI_Send(&scope, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
				MPI_Send(&points[contor].phi, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
				MPI_Send(&points[contor].p, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
				contor++;
			}

			int unfinished_processes = world_size - 1;
			//While all points are not processed or all other processes are not finished 
			while (contor < nr_points || unfinished_processes != 0) {
				double result;
				MPI_Status status;
				MPI_Recv(&result, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
					&status);

				if (processes[status.MPI_SOURCE] == 0) {
					processes[status.MPI_SOURCE]++;
					if (best_score < result) {
						best_score = result;
						is_phi = false;
						rank = status.MPI_SOURCE;
					}
				}
				else {
					if (rank == status.MPI_SOURCE) {
						if (is_phi == false) {
							best_phi = result;
							is_phi = true;
						}
						else {
							best_p = result;
						}
					}
					processes[status.MPI_SOURCE]++;
					if (processes[status.MPI_SOURCE] == 3) {
						contor++;
						//If there are unproccesed poins, send to the same process
						if (contor < nr_points) {
							processes[status.MPI_SOURCE] = 0;
							double scope = 1;
							MPI_Send(&scope, 1, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
							MPI_Send(&points[contor].phi, 1, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
							MPI_Send(&points[contor].p, 1, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
						}
						//Stop the other process
						else {
							double scope = 0;
							MPI_Send(&scope, 1, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
							unfinished_processes--;
						}
					}
				}
			}
		}
		else {
			for (int process = 1; process < world_size; process++) {
				//Send first points to processes
				if (contor < nr_points) {
					double scope = 1;
					MPI_Send(&scope, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
					MPI_Send(&points[contor].phi, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
					MPI_Send(&points[contor].p, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
					contor++;
				}
				//Stop the other procceses
				else {
					double scope = 0;
					MPI_Send(&scope, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
					contor++;
				}
			}
			
			//Receive messages from unknown process
			int process = 1;
			while (process <= nr_points) {
				double result;
				MPI_Status status;
				MPI_Recv(&result, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
					&status);

				if (processes[status.MPI_SOURCE] == 0) {
					processes[status.MPI_SOURCE]++;
					if (best_score < result) {
						best_score = result;
						is_phi = false;
						rank = status.MPI_SOURCE;
					}
				}
				else {
					if (rank == status.MPI_SOURCE) {
						if (is_phi == false) {
							best_phi = result;
							is_phi = true;
						}
						else {
							best_p = result;
						}
					}
					processes[status.MPI_SOURCE]++;
					if (processes[status.MPI_SOURCE] == 3) {
						double scope = 0;
						MPI_Send(&scope, 1, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
						process++;
					}

				}
			}
			delete[] processes;
		}
		if(FULL_OUTPUT){
			printf("BP %.2f %.2f %.2f\n",best_score, best_phi, best_p);
		}
		if(DEBUG_OUTPUT){
			printf("Finest climber at %.2f meters : (%.2f, %.2f)\n", best_score, best_phi, best_p);	
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

	//get number of starting points
	int nr_points = 0;
	if(argc >= 2){
		sscanf(argv[1], "%d", &nr_points);
	}
	

	//Create points where climbers can start
	Point points[20];
	for (int i = 0; i < nr_points; i++) {
		double phi = rand() % 20;
		double p = rand() % 20;
		points[i] = { phi, p };
	}
	
	//Sequential implementation for hill climbing
	if (world_rank == 0) {
		double sequential_time_start, sequential_time_end;
		
		sequential_time_start = MPI_Wtime();
		sequential_climbing(points, nr_points);
		sequential_time_end = MPI_Wtime();

		if(DEBUG_OUTPUT){
			printf("Execution time for sequential climbing: %.4f seconds\n", sequential_time_end - sequential_time_start);	
			printf("\nParallel climbing:\n");
		}
		if(FULL_OUTPUT){
			printf("FS %.4f\n", sequential_time_end - sequential_time_start);  // Sequential algo duration
		}
	}


	MPI_Barrier(MPI_COMM_WORLD);

	//Parallel implementation for hill climbing
	double parallel_time_start, parallel_time_end;
	if (world_rank == 0) {
		parallel_time_start = MPI_Wtime();
	}
	parallel_climbing(world_rank, world_size, points, nr_points);

	if (world_rank == 0) {
		parallel_time_end = MPI_Wtime();
		if(DEBUG_OUTPUT){
			printf("Execution time for parallel climbing: %.4f seconds\n", parallel_time_end - parallel_time_start);	
		}
		
	}

	MPI_Barrier(MPI_COMM_WORLD);
	
	//Perfect parallel implementation for hill climbing
	double parallel_time_start_perfect, parallel_time_end_perfect;
	if (world_rank == 0) {
		if(DEBUG_OUTPUT){
			printf("\nPerfect parallel climbing:\n");	
		}
		parallel_time_start_perfect = MPI_Wtime();
	}
	perfect_parallel_climbing(world_rank, world_size, points, nr_points);

	if (world_rank == 0) {
		parallel_time_end_perfect = MPI_Wtime();
		if(FULL_OUTPUT){
			printf("FP %.4f\n", parallel_time_end_perfect - parallel_time_start_perfect); // Parralel algo duration
		}
		if(DEBUG_OUTPUT){
			printf("Execution time for perfect parallel climbing: %.4f seconds\n", parallel_time_end_perfect - parallel_time_start_perfect);	
		}
		
	}

	MPI_Finalize();
}
