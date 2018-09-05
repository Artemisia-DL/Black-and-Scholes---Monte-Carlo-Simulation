# Disclamer

This project was done while studing Heston Model for personal interest. Therefore the code leaves room to many improvements especially 
on the side of optimization and speed. The main aim was didatic, hence it may lack of elegance and precision, but I hope it maybe usefull
to anyone and I encourage the few interested to improve it.

# Black and Schole - Monte Carlo Simulation

A first experiment of pricing Single Barrier Option under BS model with Numerical Simulation. 
It includes: 

    - Sampling form Standard Normal
    - Sampling from T-Student distribution ( useless, whatever, I left it.. )
    - Many goodlooking graphs
    - User Friendly approach ( No, really, I tried my best )
    - Built-in Analysis of Convergence and Accuracy ( Experimental, really experimental)
    - Sum-up of performance

The simulation function runs simulations while periodically checking for convergence accuracy level (more on this in the code notes).
Moreover, it saves ALL the paths generated and some statistic about each path ( such as min, max price which are useful in case of sigle
barrier pricing ). Note that this feature makes the algorithm computationaly slow and it can be easily avoid for Barrier Pricing,
nevertheless it is usuful during tests, hence is left.

Once the accuracy level is reached or the usuer stop the simulation the fucntion also generates many plots to better visualize what
went on during the simulation. Some of these plots are base on the paths data ( see above ) that were collected and saved. 
Again, not useful for the mere sake of the simualtion and easily optimizable.

