"""
Parallel normalization for large datasets.
"""
from multiprocessing import Pool, cpu_count
import pandas as pd
from normalize import normalize_name, name_tokens_sorted, normalize_address, extract_numeric_tokens, extract_landmark


def _normalize_chunk(args):
    """Process a chunk of data."""
    names, addrs, start_idx = args
    results = []
    for i, (name, addr) in enumerate(zip(names, addrs)):
        results.append({
            'idx': start_idx + i,
            'normalized_name': normalize_name(name),
            'name_sorted': name_tokens_sorted(name),
            'normalized_address': normalize_address(addr),
            'address_numbers': extract_numeric_tokens(addr),
            'landmark': extract_landmark(addr)
        })
    return results


def add_normalized_columns_parallel(df, name_col="business_name", addr_col="business_address", n_jobs=-1, chunk_size=50000):
    """
    Parallel version of add_normalized_columns for large datasets.
    
    Args:
        df: Input DataFrame
        name_col: Column name for business names
        addr_col: Column name for addresses
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        chunk_size: Records per chunk
    """
    if n_jobs == -1:
        n_jobs = cpu_count()
    
    df = df.copy()
    names = df[name_col].values
    addrs = df[addr_col].values
    n = len(df)
    
    # Create chunks
    chunks = []
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunks.append((names[i:end], addrs[i:end], i))
    
    print(f"Processing {n:,} records in {len(chunks)} chunks using {n_jobs} workers...")
    
    # Process in parallel
    with Pool(n_jobs) as pool:
        results = pool.map(_normalize_chunk, chunks)
    
    # Flatten results
    all_results = [item for chunk_result in results for item in chunk_result]
    
    # Add columns to dataframe
    for col in ['normalized_name', 'name_sorted', 'normalized_address', 'address_numbers', 'landmark']:
        df[col] = [r[col] for r in all_results]
    
    return df
