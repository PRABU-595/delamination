import scipy.io
import numpy as np

FILE_PATH = r"C:\Users\iampr\Desktop\MY PAPERS\delamination-ml-project\data\raw\NASA_CFRP\2. Composites\L2_S11_F\PZT-data\L2S11_0_2_1.mat"

print(f"🕵️‍♂️ Inspecting Real Data File: {FILE_PATH}")

try:
    mat = scipy.io.loadmat(FILE_PATH)
    print("✅ File loaded successfully (Valid MATLAB format).")
    
    if 'coupon' not in mat:
        print("❌ 'coupon' key not found.")
    else:
        coupon = mat['coupon']
        print(f"\n🧬 Structure 'coupon' found. Shape: {coupon.shape}")
        print(f"   Type: {type(coupon)}")
        print(f"   Dtype: {coupon.dtype}")
        
        # Just dump the first element's fields if possible
        try:
             print(f"   First Element: {coupon[0,0]}")
        except:
             print("   Could not print element.")

        # DEBUG: Print all available fields in coupon
        if hasattr(coupon.dtype, 'names') and coupon.dtype.names:
            print(f"   🔑 Available Fields: {coupon.dtype.names}")
            
        # Try to find any data-like field
        data_field = None
        # Valid keys found in dump: 'path_data', 'straingage_data'
        for key in ['path_data', 'straingage_data', 'PZT_data']:
            if key in coupon.dtype.names:
                data_field = key
                print(f"\n   ✓ Found structure: '{data_field}'")
                inner = coupon[0,0][data_field]
                
                # Check if it's a structural array
                if inner.dtype.names:
                    print(f"     Inner keys: {inner.dtype.names}")
                    
                    # Look for signal or data inside
                    for sub_key in inner.dtype.names:
                        val = inner[0,0][sub_key]
                        
                        # Handle nested cells or arrays
                        if hasattr(val, 'shape') and val.size > 100:
                            print(f"     👉 DATA FOUND in '{sub_key}' | Shape: {val.shape}")
                            
                            # Flatten and print stats
                            flat = val.flatten()
                            
                            # If it's an object array, try to get the first element (common in MATLAB cells)
                            if flat.dtype == object and flat.size > 0:
                                try:
                                    flat = flat[0].flatten()
                                except: pass
                                
                            if np.issubdtype(flat.dtype, np.number):
                                print("\n     📊 FIRST 10 VALUES:")
                                print(f"     {flat[:10]}")
                                print(f"     Mean: {np.mean(flat):.4f} | Var: {np.var(flat):.4f}")
                                if np.var(flat) > 0:
                                     print("     ✅ STATUS: VALID EXPERIMENTAL SIGNAL VERIFIED.")
                print("   ------------------------------------------------")

except Exception as e:
    print(f"❌ ERROR: {e}")
