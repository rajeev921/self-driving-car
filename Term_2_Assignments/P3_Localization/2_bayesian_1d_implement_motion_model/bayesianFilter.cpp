//============================================================================
// Name        : bayesianFilter.cpp
// Version     : 1.0.0
// Copyright   : MBRDNA, Udacity
//============================================================================

#include "bayesianFilter.h"
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

//constructor:
bayesianFilter::bayesianFilter() {

    //TODO add is_initialized to header file
    //set initialization to false:
    // NOTE: helps us set up the initial believe state
	is_initialized_ = false;

    //TODO add control_std to header file
	//set standard deviation of control to 1.0f:
	control_std = 1.0f;

	//define size of believe, same size as map
	bel_x.resize(100,0);
	
	//TODO add bel_x_init to header file
	// NOTE: helps us not overwrite believe during
	// the motion calculation
	bel_x_init.resize(100, 0); 

}

//de-constructor:
bayesianFilter::~bayesianFilter() {

}

void bayesianFilter::process_measurement(const MeasurementPackage &measurements,
        						             const map &map_1d,
                                         help_functions &helpers){

	/******************************************************************************
	 *  Set init belief of state vector:
	 ******************************************************************************/
	 if(!is_initialized_){

		//TODO: run over map, all map_1d.lanmark_list values:
		for (int l=0; l< map_1d.landmark_list.size(); ++l){

			//TODO: get landmark l from map 

            //define landmark:
			map::single_landmark_s landmark_temp;
			//get landmark from map:
			landmark_temp = map_1d.landmark_list[l];

			//check, if landmark position x fits in map [0,100]:
			if(landmark_temp.x_f > 0 && landmark_temp.x_f < bel_x_init.size() ){
			    
			    //cast float to int:
				int position_x = int(landmark_temp.x_f) ;
				//set belief to 1:
				bel_x_init[position_x]   = 1.0f;
				bel_x_init[position_x-1] = 1.0f;
				bel_x_init[position_x+1] = 1.0f;

			} //end if
		}//end for

	//TODO: normalize initial believe * use help_function.h for reference
	bel_x_init = helpers.normalize_vector(bel_x_init);

	//set initial flag to true:
	is_initialized_ = true;
	
	}//end if
	
	/******************************************************************************
	 *  motion model and observation update
	******************************************************************************/
	std::cout <<"-->motion model for state x ! \n" << std::endl;

	//get current observations and control information:
	MeasurementPackage::control_s     controls = measurements.control_s_;
	MeasurementPackage::observation_s observations = measurements.observation_s_;

	//run over all bel_x values (index represents the pose in x!):
	for (int i=0; i< bel_x.size(); ++i){

        float pose_i = float(i) ;
        
		/**************************************************************************
		 *  posterior for motion model
		**************************************************************************/
        
        // motion posterior:
        // used to set up the convlution
		float posterior_motion = 0.0f;

		//loop over state space x_t-1 * same size as bel_x (Perform Convolution):
		for (int j=0; j< bel_x.size(); ++j){
			
			//TODO: Calculate transition probabilites using helpers.normpdf()
			// x: difference between bel_x index and state space index
			// mu: the movement from controls defined above
			// std: defined eariler
			
			float pose_j = float(j) ;
			
			
            float distance_ij = pose_i-pose_j;			

			//TODO: Calculate motion model
			// ADD the transition prob multiplied by the intial believe 
			// at state space index
			
			//transition probabilities:
			float transition_prob = helpers.normpdf(distance_ij,
                    								controls.delta_x_f,
                    								control_std) ;

		    //motion model:
			posterior_motion += transition_prob*bel_x_init[j];
		}

		//TODO: update = motion_model 
	    // set as the posterior_motion
		bel_x[i] = posterior_motion ;
	};
	
		//TODO: normalize bel_x:
        bel_x = helpers.normalize_vector(bel_x);		

		//TODO: set initial believe to bel_x for next time
		//set bel_x to bel_init:
		bel_x_init = bel_x;
}; 
		
	
