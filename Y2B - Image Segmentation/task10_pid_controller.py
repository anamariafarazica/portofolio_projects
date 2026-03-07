import numpy as np

class PIDController:
    """
    Proportional-Integral-Derivative (PID) Controller
    
    A PID controller continuously calculates an error value as the difference 
    between a desired setpoint and a measured process variable, and applies a 
    correction based on proportional, integral, and derivative terms.
    
    Attributes:
        Kp (float): Proportional gain - determines reaction to current error
        Ki (float): Integral gain - determines reaction to accumulated error
        Kd (float): Derivative gain - determines reaction to rate of error change
        previous_error (float): Error from previous iteration (for derivative calculation)
        integral (float): Accumulated error over time (for integral calculation)
    """
    
    def __init__(self, Kp, Ki, Kd):
        """
        Initialize PID controller with gain parameters
        
        Args:
            Kp (float): Proportional gain
            Ki (float): Integral gain
            Kd (float): Derivative gain
        """
        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.Kd = Kd  # Derivative gain
        
        # Initialize state variables
        self.previous_error = 0  # Previous error for derivative calculation
        self.integral = 0        # Accumulated error for integral calculation
    
    def compute(self, target, current, dt):
        """
        Compute PID control signal based on current state
        
        The PID output is calculated as:
        output = Kp * error + Ki * integral(error) + Kd * derivative(error)
        
        Args:
            target (float): Desired setpoint/target value
            current (float): Current measured value
            dt (float): Time step since last update (seconds)
            
        Returns:
            float: Control signal (velocity command for robot)
        """
        # Calculate current error
        error = target - current
        
        # Integral term: accumulate error over tim
        self.integral += error * dt
        
        # Derivative term: rate of change of error
        derivative = (error - self.previous_error) / dt

        # PID output
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        # Store error for next iteration's derivative calculation
        self.previous_error = error
        
        return output
    
    def reset(self):
        """
        Reset controller state (integral and derivative terms)
        """
        self.previous_error = 0
        self.integral = 0