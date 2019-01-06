#include <iostream>
#include <cstdio>
#include <limits>
#include <cmath>

#define DEBUG_EPOCHS false
#define DEBUG_CANDIDATES false
#define DEBUG_LAST_IMPROV false

using namespace std;

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

double* hillClimb(double *finalScore){
	// number of iterations with no moves before we conclude that we converged
	int burnoutEpochs = 25; 
	double stepSize[2] = {1, 1};
	double acceleration = 1.2;	// same acceleration for both dimensions ??? 
	double epsilon = 0.1;

	// define initial starting point
	double *currentPoint = new (nothrow) double[2];
	if(!currentPoint){
		return NULL;
	}
	for(int i = 0; i < 2; ++i){
		currentPoint[i] = 0;
	}
	
	int epoch = 1;
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
				if(DEBUG_LAST_IMPROV){
					printf("Last improvement : %.2f\n", lastImprovement);	
				}
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



int main(int argc, char const *argv[])
{
	double score;
	double *climber = hillClimb(&score);
	printf("Finest climber at %.2f meters : (%.2f, %.2f)\n", score, climber[0], climber[1]);
	return 0;
}