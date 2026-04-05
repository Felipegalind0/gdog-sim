import numpy as np


class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.0, output_limit=None, integral_limit=None):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.setpoint = float(setpoint)
        self.output_limit = output_limit
        self.integral_limit = integral_limit

        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

        # Live telemetry for debug visualization.
        self.last_measurement = 0.0
        self.last_dt = 0.0
        self.last_error = 0.0
        self.last_derivative = 0.0
        self.last_p_term = 0.0
        self.last_i_term = 0.0
        self.last_d_term = 0.0
        self.last_output_unclipped = 0.0
        self.last_output = 0.0
        self.last_was_clipped = False

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

        self.last_measurement = 0.0
        self.last_dt = 0.0
        self.last_error = 0.0
        self.last_derivative = 0.0
        self.last_p_term = 0.0
        self.last_i_term = 0.0
        self.last_d_term = 0.0
        self.last_output_unclipped = 0.0
        self.last_output = 0.0
        self.last_was_clipped = False

    def update(self, measurement, dt):
        measurement = float(measurement)
        dt = max(float(dt), 1e-6)
        error = self.setpoint - measurement

        self.integral += error * dt
        if self.integral_limit is not None:
            integral_abs_limit = abs(float(self.integral_limit))
            self.integral = float(np.clip(self.integral, -integral_abs_limit, integral_abs_limit))

        if not self.initialized:
            derivative = 0.0
            self.initialized = True
        else:
            derivative = (error - self.prev_error) / dt

        self.prev_error = error
        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * derivative
        output = p_term + i_term + d_term
        output_unclipped = float(output)
        was_clipped = False

        if self.output_limit is not None:
            output_abs_limit = abs(float(self.output_limit))
            output = float(np.clip(output, -output_abs_limit, output_abs_limit))

            # Tiny epsilon avoids false positives from fp rounding noise.
            was_clipped = abs(output - output_unclipped) > 1e-10

        self.last_measurement = measurement
        self.last_dt = dt
        self.last_error = float(error)
        self.last_derivative = float(derivative)
        self.last_p_term = float(p_term)
        self.last_i_term = float(i_term)
        self.last_d_term = float(d_term)
        self.last_output_unclipped = output_unclipped
        self.last_output = float(output)
        self.last_was_clipped = bool(was_clipped)

        return float(output)
