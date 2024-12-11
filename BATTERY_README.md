# Battery Simulation: S/PCNF/CNT Composite Material

This simulation models the electrochemical performance of a Sulfur/Porous Carbon Nanofiber/Carbon Nanotube (S/PCNF/CNT) composite material in a lithium-sulfur battery configuration.

## Physical Model Components

### 1. Capacity Fade Model

The capacity fade is modeled using two main components:

```python
base_fade = initial_capacity * exp(-decay_rate * cycle)
cnt_stabilization = 1 + 0.1 * cnt_content * (1 - exp(-cycle/20))
capacity = base_fade * cnt_stabilization
```

Where:
- `initial_capacity`: Starting capacity (default: 900 mAh/g)
- `decay_rate`: Rate of capacity loss (0.002 per cycle)
- `cnt_content`: Carbon nanotube content (15%)
- `cnt_stabilization`: CNT's contribution to capacity retention

### 2. Voltage Profile

The voltage profile incorporates the characteristic two-plateau behavior of Li-S batteries:

```python
plateau1 = 2.3 + 0.1 * sin(state_of_charge * π)  # Upper plateau
plateau2 = 2.1 + 0.05 * sin(state_of_charge * π) # Lower plateau
```

Key voltage characteristics:
- Upper plateau: ~2.3V (Li2S8 → Li2S4)
- Lower plateau: ~2.1V (Li2S4 → Li2S)
- Voltage range: 1.7V - 2.8V

### 3. Material Parameters

The simulation includes specific material parameters for S/PCNF/CNT:
- Sulfur loading: 65%
- CNT content: 15%
- Operating temperature: 298K
- C-rate: 0.2C

## Simulation Features

### 1. Cycle Simulation
- Calculates charge/discharge curves for each cycle
- Models state of charge (SoC) evolution
- Incorporates both charge and discharge voltage profiles
- Accounts for capacity fade over cycling

### 2. Performance Metrics
- Capacity retention over cycles
- Voltage efficiency
- Charge-discharge voltage hysteresis
- Energy density evolution

### 3. Visualization
The simulation provides several visualization options:
1. Capacity fade curve over cycles
2. Voltage profiles for:
   - First cycle
   - Mid-cycle
   - Final cycle
3. Interactive plots with:
   - Toggleable curves
   - Hover information
   - Customizable view

## Usage

### Basic Usage

```python
from battery_simulation import run_battery_simulation

# Run with default parameters
fig = run_battery_simulation()

# Run with custom parameters
fig = run_battery_simulation(
    initial_capacity=900,  # mAh/g
    cycles=100            # number of cycles
)
```

### Web Interface Parameters

1. Initial Capacity:
   - Range: 100-2000 mAh/g
   - Step: 10 mAh/g
   - Default: 900 mAh/g

2. Number of Cycles:
   - Range: 10-1000 cycles
   - Step: 10 cycles
   - Default: 100 cycles

## Mathematical Models

### 1. Capacity Fade

The capacity fade model combines exponential decay with CNT stabilization:

```
C(n) = C₀ * exp(-αn) * [1 + β(1 - exp(-n/τ))]

Where:
C(n) = Capacity at cycle n
C₀ = Initial capacity
α = Decay rate (0.002)
β = CNT stabilization factor (0.1 * CNT_content)
τ = Stabilization time constant (20 cycles)
```

### 2. Voltage Profile

The voltage profile is modeled using a piecewise function:

```
V(SoC) = {
    2.3 + 0.1*sin(πSoC)  for SoC > 0.5
    2.1 + 0.05*sin(πSoC) for SoC ≤ 0.5
}

Where:
SoC = State of Charge (0 to 1)
```

## Implementation Details

### Key Classes and Methods

1. `BatteryModel` class:
   - Initializes simulation parameters
   - Handles cycle-by-cycle calculations
   - Manages material properties

2. `capacity_fade_model` method:
   - Calculates capacity for each cycle
   - Incorporates degradation mechanisms
   - Accounts for CNT stabilization

3. `voltage_profile` method:
   - Generates voltage curves
   - Models two-plateau behavior
   - Handles charge/discharge profiles

4. `create_visualization` function:
   - Generates interactive plots
   - Handles data formatting
   - Creates multiple visualization layers

## Limitations and Assumptions

1. Simplified Model Assumptions:
   - Uniform temperature distribution
   - Ideal electrolyte conditions
   - Perfect electrical contact
   - Homogeneous material distribution

2. Not Included:
   - Temperature effects on degradation
   - Detailed SEI formation
   - Shuttle effect modeling
   - Local concentration variations

3. Model Limitations:
   - Simplified voltage profile
   - Basic capacity fade mechanism
   - Idealized CNT stabilization effect 