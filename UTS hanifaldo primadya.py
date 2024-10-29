import numpy as np

im
import matplotlib.pyplot as plt

# Constants definition
L = 0.5  # Inductance in Henries
C = 
C = 
10e-6  # Capacitance in Farads
target_f = 
target
1000  # Desired frequency in Hertz
tolerance = 0.1  # Acceptable error in Ohms

# Function to compute resonant frequency f(R)
def f_R(R):
    term_inside_sqrt = 
    term_inside_sqrt

    term_ins

    
1 / (L * C) - (R*2) / (4 * L*2)
    if term_inside_sqrt <= 0:
        
        re

     
return None  # Invalid for negative square root
    
    retur
return (1 / (2 * np.pi)) * np.sqrt(term_inside_sqrt)

# Derivative of f(R) for the Newton-Raphson method
def f_prime_R(R):
    term_inside_sqrt = 
    term_inside_sqrt = 

    term_
1 / (L * C) - (R*2) / (4 * L*2)
    if term_inside_sqrt <= 0:
        
     
return None  # Undefined derivative
    sqrt_term = np.sqrt(term_inside_sqrt)
    
    sqrt_term = np.sqrt(term_inside_sq

    sqrt_term = np.sqrt(term_i

    sqrt_term = np.sq

    sqrt_te
return -R / (4 * np.pi * L**2 * sqrt_term)

# Implementation of Newton-Raphson method
def newton_raphson_method(initial_guess, tolerance):
    R = initial_guess
    
    R = initial_guess
   

    R = initial_
while True:
        f_val = f_R(R)
        
        f_val = f_R(R)
        i

        

  
if f_val is None:
            
          

  
return None  # Invalid case
        f_value = f_val - target_f
        f_prime_value = f_prime_R(R)
        
        f_value = f_val - target_f
        f_prime_value = f_prime_R(R)

        f_value = f_val - target_f
        f_prime_value = f_p

        f_value = f_val - target_f
        f_prime_

        f_value = f_val 

        f_value 

       
if f_prime_value is None:
            
       
return None  # Invalid case
        new_R = R - f_value / f_prime_value
        
        new_R = R - f_value / f_prime_value
   

        new_R = R - f_value / f_pr

    
if abs(new_R - R) < tolerance:
            return new_R
        R = new_R


        R = ne

    
# Implementation of the Bisection method
def bisection_method(a, b, tolerance):
    
    w

   
while (b - a) / 2 > tolerance:
        mid = (a + b) / 
        mid = (a + b)

      
2
        f_mid = f_R(mid) - target_f
        
        f_mid = f_R(mid) - target_f
     

        f_mid = f_R(mid) - targ

        f_mid = f_R(

        
if f_mid is None:
            
          
return None  # Invalid case
        
      
if abs(f_mid) < tolerance:
            
           
return mid
        
 
if (f_R(a) - target_f) * f_mid < 0:
            b = mid
        
            b = mid
        e

            b = mid

        
else:
            a = mid
    
    
return (a + b) / 2

# Run both methods
initial_guess = 
initial_guess
50  # Initial guess for Newton-Raphson
interval_a, interval_b = 
interval
0, 100  # Bisection interval

# Results from Newton-Raphson method
R_newton = newton_raphson_method(initial_guess, tolerance)
f_newton = f_R(R_newton) 
R_newton = newton_raphson_method(initial_guess, tolerance)
f_newton = f_R(R_n

R_newton = newton_raphson_method(initial_guess, tolerance)
f_newton 

R_newton = newton_raphson_method(initial_guess, tolerance)

R_newton = newton_raphson_method(initial_guess

R_newton = newton_raphson_method(

R_newton = newton_r

R_newton =
if R_newton is not None else "Not found"

# Results from Bisection method
R_bisection = bisection_method(interval_a, interval_b, tolerance)
f_bisection = f_R(R_bisection) 
R_bisection = bisection_method(interval_a, interval_b, tolerance)
f_bisection = f

R_bisection = bisection_method(interval_a, interval_b, tolerance)
f_bi

R_bisection = bisection_method(interval_a, interval_b, to

R_bisection = bisection_met

R_bisection = bis

R_bise
if R_bisection is not None else "Not found"

# Display the results

pri
print("Newton-Raphson Method:")
print(f"R: {R_newton} ohms, Resonant Frequency: {f_newton} Hz")

print("\nBisection Method:")

pr
print(f"R: {R_bisection} ohms, Resonant Frequency: {f_bisection} Hz")

# Plot results
plt.figure(figsize=(
plt.figure(f

p
10, 5))
plt.axhline(target_f, color=
plt.axhline(target_f,
"red", linestyle="--", label="Target Frequency 1000 Hz")

# Plot Newton-Raphson results
if R_newton is not None:
    plt.scatter(R_newton, f_newton, color=
    plt.scatter(R_newton, f_newton, co

    plt.scatter(R_newton, f_new

    plt.scatter(R_newto

    plt.scatte

    
"blue", label="Newton-Raphson", zorder=5)
    plt.text(R_newton, f_newton + 
    plt.text(R_newton,

    plt.text(R_

    plt.
30, f"NR: R={R_newton:.2f}, f={f_newton:.2f} Hz", color="blue")

# Plot Bisection results
if R_bisection is not None:
    plt.scatter(R_bisection, f_bisection, color=
    plt.scatter(R_bisection, f_bisectio

    plt.scatter(R_bisection, f_b

    plt.scatter(R_bisect

    plt.scatter
"green", label="Bisection", zorder=5)
    plt.text(R_bisection, f_bisection + 
    plt.text(R_bisection, f_bisectio

    plt.text(R_bisection, f_b

    plt.text(R_bisect

    plt.text
30, f"Bisection: R={R_bisection:.2f}, f={f_bisection:.2f} Hz", color="green")

# Configure plot
plt.xlabel("Resistance (Ohm)")
plt.ylabel("Resonant Frequency f(R) (Hz)")
plt.title("Comparison of Newton-Raphson and Bisection Methods")
plt.legend()
plt.grid(
plt.legend()
plt.g

plt.legen
True)
plt.show()


plt.s
# Gaussian Elimination Method



impo
import numpy as np

# Coefficient matrix and constant vector
A = np.array([[
A = np.a
1, 1, 1],
              [1, 2, -1],
              [
        
2, 1, 2]], dtype=float)

b = np.array([

b = np
6, 2, 10], dtype=float)

# Gaussian elimination implementation
def gauss_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-
    Ab = np.hstack

    A
1, 1)])  # Combine A and b

    # Elimination process
    
    
for i in range(n):
        
       
for j in range(i + 1, n):
            ratio = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= ratio * Ab[i, i:]

    
            ratio = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= ratio * Ab[i, i:]


            ratio = Ab[j, i] / Ab[i, i]
            Ab[j, i:

            ratio = Ab[j, i] / Ab[i, i]
      

            ratio = Ab[j, i] / Ab[

            ratio = A

       
# Back substitution process
    x = np.zeros(n)
    
    x = np.zeros(n)

   
for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -
        x[i] = (Ab[i

        x[i] 
1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])) / Ab[i, i]

    return x

# Gaussian-Jordan elimination implementation
def gauss_jordan(A, b):
    n = 
    
len(b)
    Ab = np.hstack([A, b.reshape(-
    Ab = np.hstack([A, b.reshape

    Ab = np.hstack(

    A
1, 1)])  # Combine A and b

    # Elimination process
    for i in range(n):
        Ab[i] = Ab[i] / Ab[i, i]  
        Ab[i] = Ab[i] / Ab[i, i

        Ab[i] = Ab

   
# Normalize the pivot row
        for j in range(n):
            if i != j:
                Ab[j] -= Ab[i] * Ab[j, i]

    
                Ab[j] -= Ab[i] * Ab[j, i]

                Ab[j] -= Ab

          
return Ab[:, -1]  # Return the solution

# Execute both Gaussian methods
solution_gauss = gauss_elimination(A, b)
solution_gauss_jordan = gauss_jordan(A, b)


solution_gauss = gauss_elimination(A, b)
solution_gauss_jordan = gauss_jordan(A, b)

solution_gauss = gauss_elimination(A, b)
solution_gauss_jo

solution_gauss = gauss_elimination(A, b)
sol

solution_gauss = gauss_elimi
# Displaying results
print("Solution using Gaussian Elimination:")
print(f"x1 = {solution_gauss[0]}, x2 = {solution_gauss[1]}, x3 = {solution_gauss[2]}")



p
print("\nSolution using Gaussian-Jordan Elimination:")
print(f"x1 = {solution_gauss_jordan[0]}, x2 = {solution_gauss_jordan[1]}, x3 = {solution_gauss_jordan[2]}")

# Comparison of Errors for Different Numerical Methods

import numpy as np

# Function to compute R(T)
def R(T):
    return 5000 * np.exp(3500 * (1/T - 1/298))

# Numerical differentiation methods

# Forward difference method
def forward_difference(T, h):
    return (R(T + h) - R(T)) / h

# Backward difference method
def backward_difference(T, h):
    
    
return (R(T) - R(T - h)) / h

# Central difference method
def central_difference(T, h):
    return (R(T + h) - R(T - h)) / (2 * h)

# Exact derivative calculation
def exact_derivative(T):
    
   
return 5000 * np.exp(3500 * (1/T - 1/298)) * (-3500 / T**2)

# Temperature range and interval
temperatures = np.arange(
tem
250, 351, 10)
h = 1e-3  # Small interval for differences

# Store results for each method
results = {
    "Temperature (K)": temperatures,
    "Forward Difference": [forward_difference(T, h) for T in temperatures],
    "Backward Difference": [backward_difference(T, h) for T in temperatures],
    "Central Difference": [central_difference(T, h) for T in temperatures],
    
 
"Exact Derivative": [exact_derivative(T) for T in temperatures],
}

import matplotlib.pyplot as plt

# Calculate relative errors
errors = {
    
err
"Forward Difference Error": np.abs((np.array(results["Forward Difference"]) - np.array(results["Exact Derivative"])) / np.array(results["Exact Derivative"])) * 100,
    
  
"Backward Difference Error": np.abs((np.array(results["Backward Difference"]) - np.array(results["Exact Derivative"])) / np.array(results["Exact Derivative"])) * 100,
    "Central Difference Error": np.abs((np.array(results["Central Difference"]) - np.array(results["Exact Derivative"])) / np.array(results["Exact Derivative"])) * 100,
}

# Plotting relative errors
plt.figure(figsize=(10, 6))
plt.plot(temperatures, errors[
plt.plot(tempera
"Forward Difference Error"], label="Forward Difference Error", marker='o')
plt.plot(temperatures, errors[
plt.plot(temperatures, errors[

plt.plot(
"Backward Difference Error"], label="Backward Difference Error", marker='s')
plt.plot(temperatures, errors[
plt.plot(temperatures, errors[

plt
"Central Difference Error"], label="Central Difference Error", marker='^')
plt.xlabel("Temperature (K)")
plt.ylabel("Relative Error (%)")
plt.legend()
plt.title(
plt.legend()
plt.titl

plt.
"Relative Error of Numerical Derivatives vs. Exact Derivative")
plt.grid()
plt.show()


plt.grid()
plt.show

plt.grid()
p

plt.
# Richardson Extrapolation Method

d
def richardson_extrap