# CarND-Controls-MPC
Self-Driving Car Engineer Nanodegree Program

---

PROJECT WRITEUP

Model Overview

After implementing PID control in a previous project, we will now use MPC (Model Predictive Control) in the current project to improve and build a more stable controller. This will control the steering, velocity and throttle of the car to promise an optimised predicted trajectory.

This is done by modeling the state of the vehicle and predicting the state based on actuators into the future. More specifically, this is achieved by implementing this as a cost function and trying to minimize it.
State

  1. px: x-axis vehicle position
  2. py: y-axis vehicle position
  3. psi: heading of vehicle
  4. v: speed of vehicle
  5. cte: cross track error (offset of vehicle from center of road)
  6. epsi: heading error (diff between ideal and actual heading/direction)

Actuators

  Steering
  Throttle - this combines acceleration and braking into one value

Update

The model loop runs at a frequency of every 125ms. This delay takes into account the processing time of solver as well as the latency involved in communication with the simulator.

Tuning

  N = 10 (number of predictions to make into the future, as large as possible)
  dt = 0.1 (the time gap between predictions, as small as possible)

If N is too large, it will slow down the processing without any benefit as the prediction too far into the future will be invalid. Similarly, too small a dt will require even more frequent calculations, and if the model is not able to keep up, then it will lead to instability.

Pre-processing

Incoming data from the simulator that needs to be transformed:

  Reference trajectory - x
  Reference trajectory - y
  Vehicle position - x
  Vehicle position - y
  Heading of the vehicle - psi
  Speed of the vehicle - v
  Steering
  Throttle

1. First, waypoint coordinates are transformed to global co-ordinate system:

  ptsx[i] = (shift_x * cos(0-psi) - shift_y * sin(0-psi))
  ptsy[i] = (shift_x * sin(0-psi) + shift_y * cos(0-psi))

2. Next, cross track error is found by fitting the transformed coordinates into third degree polynomial.

  fg[2 + x_start + i]    = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
  fg[2 + y_start + i]    = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
  fg[2 + psi_start + i]  = psi1 - (psi0 + v0 * delta0 / Lf * dt);
  fg[2 + v_start + i]    = v1 - (v0 + a0 * dt);
  fg[2 + cte_start + i]  = cte1 - ((f0 - y0) + v0 * CppAD::sin(epsi0) * dt);
  fg[2 + epsi_start + i] = epsi1 - ((psi0 - psides0) + v0 * delta0 / Lf * dt);

3. Re-evaluation of the state considering 125 ms delay, before passing the data to the solver.


---

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Fortran Compiler
  * Mac: `brew install gcc` (might not be required)
  * Linux: `sudo apt-get install gfortran`. Additionall you have also have to install gcc and g++, `sudo apt-get install gcc g++`. Look in [this Dockerfile](https://github.com/udacity/CarND-MPC-Quizzes/blob/master/Dockerfile) for more info.
* [Ipopt](https://projects.coin-or.org/Ipopt)
  * Mac: `brew install ipopt`
  * Linux
    * You will need a version of Ipopt 3.12.1 or higher. The version available through `apt-get` is 3.11.x. If you can get that version to work great but if not there's a script `install_ipopt.sh` that will install Ipopt. You just need to download the source from the Ipopt [releases page](https://www.coin-or.org/download/source/Ipopt/) or the [Github releases](https://github.com/coin-or/Ipopt/releases) page.
    * Then call `install_ipopt.sh` with the source directory as the first argument, ex: `sudo bash install_ipopt.sh Ipopt-3.12.1`. 
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [CppAD](https://www.coin-or.org/CppAD/)
  * Mac: `brew install cppad`
  * Linux `sudo apt-get install cppad` or equivalent.
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/self-driving-car-sim/releases).
* Not a dependency but read the [DATA.md](./DATA.md) for a description of the data sent back from the simulator.


## Basic Build Instructions


1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./mpc`.

## Tips

1. It's recommended to test the MPC on basic examples to see if your implementation behaves as desired. One possible example
is the vehicle starting offset of a straight line (reference). If the MPC implementation is correct, after some number of timesteps
(not too many) it should find and track the reference line.
2. The `lake_track_waypoints.csv` file has the waypoints of the lake track. You could use this to fit polynomials and points and see of how well your model tracks curve. NOTE: This file might be not completely in sync with the simulator so your solution should NOT depend on it.
3. For visualization this C++ [matplotlib wrapper](https://github.com/lava/matplotlib-cpp) could be helpful.

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!

More information is only accessible by people who are already enrolled in Term 2
of CarND. If you are enrolled, see [the project page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/f1820894-8322-4bb3-81aa-b26b3c6dcbaf/lessons/b1ff3be0-c904-438e-aad3-2b5379f0e0c3/concepts/1a2255a0-e23c-44cf-8d41-39b8a3c8264a)
for instructions and the project rubric.

## Hints!

* You don't have to follow this directory structure, but if you do, your work
  will span all of the .cpp files here. Keep an eye out for TODOs.

## Call for IDE Profiles Pull Requests

Help your fellow students!

We decided to create Makefiles with cmake to keep this project as platform
agnostic as possible. Similarly, we omitted IDE profiles in order to we ensure
that students don't feel pressured to use one IDE or another.

However! I'd love to help people get up and running with their IDEs of choice.
If you've created a profile for an IDE that you think other students would
appreciate, we'd love to have you add the requisite profile files and
instructions to ide_profiles/. For example if you wanted to add a VS Code
profile, you'd add:

* /ide_profiles/vscode/.vscode
* /ide_profiles/vscode/README.md

The README should explain what the profile does, how to take advantage of it,
and how to install it.

Frankly, I've never been involved in a project with multiple IDE profiles
before. I believe the best way to handle this would be to keep them out of the
repo root to avoid clutter. My expectation is that most profiles will include
instructions to copy files to a new location to get picked up by the IDE, but
that's just a guess.

One last note here: regardless of the IDE used, every submitted project must
still be compilable with cmake and make./
