#include <iostream>
#include <cstdio>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <chrono>

#include <thread>
#include <vector>
#include <future>

#define DEBUG_EPOCHS false
#define DEBUG_CANDIDATES false
#define DEBUG_LAST_IMPROV true
#define DEBUG_ITERATIONS true

using namespace std;
using namespace std::chrono;

class Data
{
    public:
        double phi;
        double p;
};

double func(double point[2]){
	double phi = point[0];
	double p = point[1];
	double result = 35000 * sin(3 * phi) * sin(2 * p)
		+ 9700 * cos(10 * phi) * cos(20 * p)
		- 800 * sin(25 * phi + 0.03 * M_PI)
		+ 550 * cos(p + 0.2 * M_PI);
	return result;
}


double candidate(int c, int acceleration){
	switch(c){
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

void printStatus(double currentPoint[2], double stepSize[2]){
	printf("Current Point : (%.2f, %.2f). Step size : (%.2f, %.2f). \n",
				currentPoint[0], currentPoint[1], stepSize[0], stepSize[1]);
}

double* hillClimb(double *finalScore, int coordX, int coordY){
	// number of iterations with no moves before we conclude that we converged
	int burnoutEpochs = 100;
	double stepSize[2] = {0.5, 0.5};
	double acceleration = 2;	// same acceleration for both dimensions ???
	double epsilon = 0.1;

	// define initial starting point
	double *currentPoint = new (nothrow) double[2];
	if(!currentPoint){
		return NULL;
	}
	currentPoint[0] = coordX;
	currentPoint[1] = coordY;

	int epoch = 1;
	int iterations = 0;
	double lastImprovement = 0;
	while(true){
		// compute initial score
		double before = func(currentPoint);
		for(int i = 0; i < 2; ++i){
			int best = -1; //best candidate
			double bestScore = - numeric_limits<double>::infinity();  // minus INF
			double currentValue = currentPoint[i];
			for(int j = 1; j <= 5; ++j){
				currentPoint[i] = currentPoint[i] + stepSize[i] * candidate(j, acceleration);
				double temp = func(currentPoint);
				if(DEBUG_CANDIDATES){
					printf("Dimension %d - candidate %d\n",i, j);
					printStatus(currentPoint, stepSize);
					printf("Value : %.2f\n",temp );
					printf("\n");
				}
				currentPoint[i] = currentValue;	//revert changes
				if(temp > bestScore){
					bestScore = temp;
					best = j;
				}

			}
			if(DEBUG_CANDIDATES){
				printf("Dimension %d - best candidate %d\n",i, best);
			}
			if(candidate(best, acceleration) == 0){
				stepSize[i] = stepSize[i] / acceleration;
			}
			else{
				currentPoint[i] = currentPoint[i] + stepSize[i] * candidate(best, acceleration);
				stepSize[i] = stepSize[i] * candidate(best, acceleration); // accelerate
			}

		}

		double now = func(currentPoint);
		double improvement = now - before;
		++iterations;
		if(DEBUG_EPOCHS){
			printf("Epoch %d !!!!!!!!!!!!!!!!!!!!\n",epoch);
			printf("Now : %.2f - improvement : %.2f \n", now, improvement );
			printStatus(currentPoint, stepSize);
		}
		if(improvement < epsilon){
			if(epoch < burnoutEpochs){
				++epoch;
			}
			else{
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

double* hill_climbing_point(promise<Data> && p, double *finalScore, int coordX, int coordY){
	// number of iterations with no moves before we conclude that we converged
	int burnoutEpochs = 100;
	double stepSize[2] = {0.5, 0.5};
	double acceleration = 2;	// same acceleration for both dimensions ???
	double epsilon = 0.1;

	// define initial starting point
	double *currentPoint = new (nothrow) double[2];
	if(!currentPoint){
		return NULL;
	}

	currentPoint[0] = coordX;
	currentPoint[1] = coordY;

	int epoch = 1;
	int iterations = 0;
	double lastImprovement = 0;
	while(true){
		// compute initial score
		double before = func(currentPoint);
		for(int i = 0; i < 2; ++i){
			int best = -1; //best candidate
			double bestScore = - numeric_limits<double>::infinity();  // minus INF
			double currentValue = currentPoint[i];
			for(int j = 1; j <= 5; ++j){
				currentPoint[i] = currentPoint[i] + stepSize[i] * candidate(j, acceleration);
				double temp = func(currentPoint);
				if(DEBUG_CANDIDATES){
					printf("Dimension %d - candidate %d\n",i, j);
					printStatus(currentPoint, stepSize);
					printf("Value : %.2f\n",temp );
					printf("\n");
				}
				currentPoint[i] = currentValue;	//revert changes
				if(temp > bestScore){
					bestScore = temp;
					best = j;
				}

			}
			if(DEBUG_CANDIDATES){
				printf("Dimension %d - best candidate %d\n",i, best);
			}
			if(candidate(best, acceleration) == 0){
				stepSize[i] = stepSize[i] / acceleration;
			}
			else{
				currentPoint[i] = currentPoint[i] + stepSize[i] * candidate(best, acceleration);
				stepSize[i] = stepSize[i] * candidate(best, acceleration); // accelerate
			}

		}

		double now = func(currentPoint);
		double improvement = now - before;
		++iterations;
		if(DEBUG_EPOCHS){
			printf("Epoch %d !!!!!!!!!!!!!!!!!!!!\n",epoch);
			printf("Now : %.2f - improvement : %.2f \n", now, improvement );
			printStatus(currentPoint, stepSize);
		}
		if(improvement < epsilon){
			if(epoch < burnoutEpochs){
				++epoch;
			}
			else{
				*finalScore = now;
				Data data;
				data.phi = currentPoint[0];
				data.p = currentPoint[1];
				p.set_value(data);
				return currentPoint;
			}
		}
		else {
			epoch = 0;
			lastImprovement = improvement;
		}

	}
}

void sequential_climbing() {
    cout << "\nSequential climbing:" << endl;
    double score;
    double bestClimber[2], bestScore;
	for(int i = 0; i < 100; i++) {
        double *climber = hillClimb(&score, rand() % 100, rand() % 100);
        if(score > bestScore) {
            bestScore = score;
            bestClimber[0] = climber[0];
            bestClimber[1] = climber[1];
        }
	}
	printf("Finest climber at %.2f meters : (%.2f, %.2f)\n", bestScore, bestClimber[0], bestClimber[1]);
}

void parallel_climbing() {
    cout << "\nParallel climbing:" << endl;
    std::vector<std::thread> threads;
    promise<Data> promises[100];
    std::vector<future<Data>> futures;

	double score;
	double best, best_climber[2], coord[2];
	for(int i = 0; i < 100; i++) {
        futures.push_back(promises[i].get_future());
        threads.push_back(std::thread(hill_climbing_point, std::move(promises[i]),&score, rand() % 100, rand() % 100));
	}

    for(auto &t : threads){
         t.join();
    }

    for(auto &future : futures) {
        Data data = future.get();
        coord[0] = data.phi;
        coord[1] = data.p;

        if(func(coord) > best) {
            best = func(coord);
            best_climber[0] = data.phi;
            best_climber[1] = data.p;
        }
    }

	printf("Finest climber at %.2f meters : (%.2f, %.2f)\n", func(best_climber), best_climber[0], best_climber[1]);
}



int main(int argc, char const *argv[])
{
    std::srand (std::time (0));
	high_resolution_clock::time_point t3 = high_resolution_clock::now();
	sequential_climbing();
    high_resolution_clock::time_point t4 = high_resolution_clock::now();

	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	parallel_climbing();
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    cout << "\nExecution sequential climbing : " << duration_cast<milliseconds>( t4 - t3 ).count() << " milliseconds.";
    cout << "\nExecution parallel climbing : " << duration_cast<milliseconds>( t2 - t1 ).count() << " milliseconds." << endl;

	return 0;
}
