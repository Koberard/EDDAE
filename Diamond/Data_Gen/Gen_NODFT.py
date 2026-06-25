import numpy as np
import scipy.ndimage
import h5py
import os

# ==============================================================================
# 1. CONFIGURATION: THE "CONTROL PANEL"
# ==============================================================================
# This section defines the physical and statistical scale of your simulation.
# - SHAPE: The resolution of the 3D voxel grid.
# - N_ELECTRONS: The total charge (integral of the density) to be conserved.
# - NUM_TRAIN/VAL: The total number of unique density volumes to generate.
# - MC_SAMPLES: Simulates VMC "effort." Lower numbers create more grain/noise.
# ==============================================================================
SHAPE = (64,64,64)  
N_ELECTRONS = 8.0   
NUM_TRAIN = 10000   
NUM_VAL = 2000      
MC_SAMPLES = 10**6  

# ==============================================================================
# 2. SYNTHETIC DFT GENERATION: THE SMOOTH REFERENCE
# ==============================================================================
def create_fake_dft(shape):
    """
    Generates a smooth baseline density representing a DFT calculation.
    It simulates 'atoms' by placing Gaussian distributions in the grid.
    
    The number of atoms is defined by the number of tuples in the 'centers' list.
    Currently, there are 3 atoms defined.
    """
    print(f"  > Generating Synthetic DFT Reference {shape}...")
    grid = np.zeros(shape, dtype=np.float32)
    
    # Each tuple below defines the (Z, Y, X) coordinates for one 'atom'.
    # To add or remove atoms, simply modify this list.
    centers = [
        (shape[0]//2, shape[1]//2, shape[2]//2),       # Atom 1: Center
        (shape[0]//3, shape[1]//2, shape[2]//2),       # Atom 2: Offset Z
        (shape[0]//2, shape[1]//2 + 15, shape[2]//2)   # Atom 3: Offset Y
    ]
    
    z, y, x = np.indices(shape)
    
    # We loop through each center and add a Gaussian cloud (width=6.0).
    # This creates the 'atomic core' signals that the neural network 
    # will eventually subtract out to focus on the correlation.
    for (z0, y0, x0) in centers:
        r2 = (z - z0)**2 + (y - y0)**2 + (x - x0)**2
        grid += np.exp(-r2 / (2 * 6.0**2)) 
        
    # PHYSICS CONSTRAINT: Real density must sum to the total number of electrons.
    # We re-normalize the grid so that np.sum(grid) == N_ELECTRONS.
    grid *= (N_ELECTRONS / np.sum(grid))
    return grid

# ==============================================================================
# 3. ELECTRON CORRELATION LOGIC: THE "SIGNAL"
# ==============================================================================
def generate_blobs(shape, scale=6):
    """
    Creates smooth, low-frequency fluctuations. In real VMC, this represents
    'Electron Correlation'—the complex ways electrons avoid each other that
    DFT misses. This is the 'signal' we want the AI to learn to find.
    """
    # 1. Create a tiny, random low-res grid.
    low_res_shape = np.maximum(np.array(shape) // scale, 1)
    field = np.random.uniform(-1.0, 1.0, size=low_res_shape)
    
    # 2. Upsample and interpolate to create smooth 'blobs' instead of grain.
    # This ensures the signal is physically coherent across multiple voxels.
    zoom = np.array(shape) / np.array(low_res_shape)
    blobs = scipy.ndimage.zoom(field, zoom, order=2)
    
    # 3. Ensure the output matches the exact SHAPE requested.
    return blobs[:shape[0], :shape[1], :shape[2]]

# ==============================================================================
# 4. DATASET FACTORY: PRODUCING (INPUT, TARGET) PAIRS
# ==============================================================================
def generate_dataset(n_items, dft_ref, n_samples):
    """
    This loop builds the Training and Validation sets.
    - X_DATA (Input): Noisy, sampled VMC-like residuals.
    - Y_DATA (Target): Clean, physically-corrected 'Ground Truth' residuals.
    """
    x_data = np.zeros((n_items, *dft_ref.shape), dtype=np.float32)
    y_data = np.zeros((n_items, *dft_ref.shape), dtype=np.float32)
    
    print(f"  > Generating {n_items} pairs using N={n_samples}...")
    
    for i in range(n_items):
        # A. Create "True VMC" Density (The Ground Truth)
        # We perturb the smooth DFT with our 'blobs' to create a new, 
        # more accurate physical state.
        blobs = generate_blobs(dft_ref.shape)
        perturbation_strength = np.random.uniform(0.10, 0.20)
        
        # True = DFT * (1 + blobs). This adds physical correlation to the atoms.
        true_rho = dft_ref * (1.0 + (perturbation_strength * blobs))
        true_rho = np.maximum(true_rho, 1e-12) # Prevent negative density
        
        # Conserve charge after adding the blobs.
        true_rho *= (N_ELECTRONS / np.sum(true_rho))
        
        # B. Create Noisy Input (Simulating the VMC Sampling Process)
        # Real VMC is 'grainy' because it uses random samples. 
        # stochastic_density() adds that Monte Carlo shot noise.
        noisy_rho = stochastic_density(true_rho, n_samples)
        
        # C. Transform to Residual Space
        # Instead of learning the density, we learn the *difference* from DFT.
        # This makes it easier for the model to see the small correlation signal.
        x_res = transform(noisy_rho, dft_ref, 'residual_noise') # Noisy Residual
        y_res = transform(true_rho, dft_ref, 'residual_noise')  # Clean Residual
        
        x_data[i] = x_res
        y_data[i] = y_res
        
        if i % 100 == 0 and i > 0:
            print(f"    ...batch {i} complete")
            
    return x_data, y_data

# ==============================================================================
# 5. MAIN EXECUTION: WORKFLOW & STORAGE
# ==============================================================================
if __name__ == "__main__":
    print("--- STARTING SYNTHETIC DATA GENERATION ---")
    
    # 1. Initialize the global reference (The 'Molecule' geometry).
    dft_ref = create_fake_dft(SHAPE)
    
    # 2. Generate the training pairs.
    print("\n[Training Data]")
    x_train, y_train = generate_dataset(NUM_TRAIN, dft_ref, MC_SAMPLES)
    
    # 3. Generate validation pairs (Independent from training).
    print("\n[Validation Data]")
    x_val, y_val = generate_dataset(NUM_VAL, dft_ref, MC_SAMPLES)
    
    # 4. Save to a compressed .npz file for later use in model.fit().
    # We save the dft_ref so we can reconstruct the full density later.
    filename = f"NO_DFT/Synthetic_Transformed_N{MC_SAMPLES}_MD.npz"
    print(f"\n>>> Saving {filename}...")
    
    np.savez(filename,
             x_train=x_train,
             y_train=y_train,
             x_val=x_val,
             y_val=y_val,
             dft_ref=dft_ref)
             
    print("Done.")