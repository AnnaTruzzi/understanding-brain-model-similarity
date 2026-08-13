import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, ttest_ind, ttest_1samp
from scipy import stats
import os

def cohens_d(group1, group2):
    """
    Calculate Cohen's d for two groups.
    
    Parameters:
    group1, group2: array-like
        The two groups to compare
        
    Returns:
    float: Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d

def test_perceptual_vs_semantic(df, measure='sum'):
    """
    Run t-tests comparing perceptual vs semantic values for each network and layer combination.
    
    Parameters:
    df: pandas DataFrame with mediation results
    measure: str, either 'sum' or 'proportion' - which column to analyze
    
    Returns:
    results: dict with test results
    """
    
    # Define the networks and layers to analyze
    networks = ['dcrandom', 'dctrained', 'alexnettrained']
    layers = ['ReLu2', 'ReLu7']
    rois = ['EVC', 'IT']
    
    # Calculate total number of tests for Bonferroni correction
    # We always have 2 one-sample tests per combination (perceptual vs 0, semantic vs 0)
    # Paired tests are only performed if both one-sample tests are significant
    # For conservative correction, assume maximum possible tests (all paired tests performed)
    one_sample_tests = len(networks) * len(layers) * len(rois) * 2  # 2 per combination
    max_paired_tests = len(networks) * len(layers) * len(rois)      # 1 per combination (maximum)
    total_tests = one_sample_tests + max_paired_tests
    alpha_original = 0.05
    alpha_corrected = alpha_original / total_tests
    
    results = {}
    results['bonferroni_info'] = {
        'total_tests': total_tests,
        'alpha_original': alpha_original,
        'alpha_corrected': alpha_corrected
    }
    
    print(f"Bonferroni correction: {total_tests} tests, corrected α = {alpha_corrected:.6f}")
    
    for roi in rois:
        results[roi] = {}
        print(f"\n{'='*60}")
        print(f"ROI: {roi}")
        print(f"{'='*60}")
        
        for network in networks:
            results[roi][network] = {}
            print(f"\nNetwork: {network}")
            print("-" * 40)
            
            for layer in layers:
                print(f"\nLayer: {layer}")
                
                # Filter data for current network, layer, and ROI
                subset = df[(df['net'] == network) & 
                           (df['layer'] == layer) & 
                           (df['ROI'] == roi)]
                
                if subset.empty:
                    print(f"  No data found for {network}, {layer}, {roi}")
                    continue
                
                # Get perceptual and semantic values
                perceptual_values = subset[subset['type'] == 'perceptual'][measure].values
                semantic_values = subset[subset['type'] == 'semantic'][measure].values
                
                if len(perceptual_values) == 0 or len(semantic_values) == 0:
                    print(f"  Insufficient data for {network}, {layer}, {roi}")
                    continue
                
                # One-sample t-tests against zero
                perceptual_t1samp = ttest_1samp(perceptual_values, 0)
                semantic_t1samp = ttest_1samp(semantic_values, 0)
                
                # Check if both distributions are significantly different from zero
                both_significant_vs_zero = (perceptual_t1samp.pvalue < alpha_corrected and 
                                          semantic_t1samp.pvalue < alpha_corrected)
                
                # Only run paired t-test if both are significantly different from zero
                if both_significant_vs_zero:
                    t_result = ttest_rel(perceptual_values, semantic_values)
                    t_stat = t_result.statistic
                    p_value = t_result.pvalue
                    df_value = t_result.df
                    # Calculate Cohen's d
                    d = cohens_d(perceptual_values, semantic_values)
                    paired_test_performed = True
                else:
                    t_stat = np.nan
                    p_value = np.nan
                    df_value = np.nan
                    d = np.nan
                    paired_test_performed = False
                
                # Calculate descriptive statistics
                perceptual_mean = np.mean(perceptual_values)
                perceptual_std = np.std(perceptual_values, ddof=1)
                semantic_mean = np.mean(semantic_values)
                semantic_std = np.std(semantic_values, ddof=1)
                
                # Store results
                results[roi][network][layer] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'df': df_value,
                    'cohens_d': d,
                    'paired_test_performed': paired_test_performed,
                    'both_significant_vs_zero': both_significant_vs_zero,
                    'perceptual_mean': perceptual_mean,
                    'perceptual_std': perceptual_std,
                    'perceptual_n': len(perceptual_values),
                    'perceptual_t1samp_stat': perceptual_t1samp.statistic,
                    'perceptual_t1samp_p': perceptual_t1samp.pvalue,
                    'perceptual_t1samp_df': perceptual_t1samp.df,
                    'semantic_mean': semantic_mean,
                    'semantic_std': semantic_std,
                    'semantic_n': len(semantic_values),
                    'semantic_t1samp_stat': semantic_t1samp.statistic,
                    'semantic_t1samp_p': semantic_t1samp.pvalue,
                    'semantic_t1samp_df': semantic_t1samp.df
                }
                
                # Print results
                print(f"  Perceptual: M = {perceptual_mean:.4f}, SD = {perceptual_std:.4f}, N = {len(perceptual_values)}")
                print(f"    One-sample t-test vs 0: t({perceptual_t1samp.df}) = {perceptual_t1samp.statistic:.4f}, p = {perceptual_t1samp.pvalue:.4f}")
                print(f"  Semantic:   M = {semantic_mean:.4f}, SD = {semantic_std:.4f}, N = {len(semantic_values)}")
                print(f"    One-sample t-test vs 0: t({semantic_t1samp.df}) = {semantic_t1samp.statistic:.4f}, p = {semantic_t1samp.pvalue:.4f}")
                
                if paired_test_performed:
                    print(f"  Paired t-test (Perceptual vs Semantic): t({df_value}) = {t_stat:.4f}, p = {p_value:.4f}")
                    print(f"  Cohen's d = {d:.4f}")
                else:
                    print(f"  Paired t-test: NOT PERFORMED (both distributions must be significantly different from zero)")
                    print(f"  Cohen's d = N/A")
                
                # Interpret Cohen's d
                if abs(d) < 0.2:
                    effect_size = "negligible"
                elif abs(d) < 0.5:
                    effect_size = "small"
                elif abs(d) < 0.8:
                    effect_size = "medium"
                else:
                    effect_size = "large"
                print(f"  Effect size: {effect_size}")
                
                # Interpret significance (both uncorrected and Bonferroni corrected)
                # For one-sample tests
                perceptual_sig_corrected = "significant" if perceptual_t1samp.pvalue < alpha_corrected else "ns"
                semantic_sig_corrected = "significant" if semantic_t1samp.pvalue < alpha_corrected else "ns"
                
                print(f"  Perceptual vs 0 (Bonferroni): {perceptual_sig_corrected}")
                print(f"  Semantic vs 0 (Bonferroni): {semantic_sig_corrected}")
                
                # For paired t-test (only if performed)
                if paired_test_performed:
                    if p_value < 0.001:
                        significance_uncorrected = "***"
                    elif p_value < 0.01:
                        significance_uncorrected = "**"
                    elif p_value < 0.05:
                        significance_uncorrected = "*"
                    else:
                        significance_uncorrected = "ns"
                    
                    if p_value < alpha_corrected:
                        significance_corrected = "significant (Bonferroni corrected)"
                    else:
                        significance_corrected = "ns (Bonferroni corrected)"
                    
                    print(f"  Paired test significance (uncorrected): {significance_uncorrected}")
                    print(f"  Paired test significance (Bonferroni): {significance_corrected}")
                else:
                    print(f"  Paired test significance: N/A (test not performed)")
    
    return results

def save_results_to_file(results, measure, output_path):
    """
    Save t-test results to a text file.
    
    Parameters:
    results: dict with test results
    measure: str, either 'sum' or 'proportion'
    output_path: str, path to save the results
    """
    
    with open(output_path, 'w') as f:
        f.write(f"T-TEST RESULTS: PERCEPTUAL vs SEMANTIC {measure.upper()}\n")
        f.write("="*70 + "\n\n")
        
        # Write Bonferroni correction information
        bonferroni_info = results['bonferroni_info']
        f.write("BONFERRONI CORRECTION:\n")
        f.write(f"Total number of tests: {bonferroni_info['total_tests']}\n")
        f.write(f"One-sample tests (always performed): {len([r for r in results.keys() if r != 'bonferroni_info']) * 2}\n")
        f.write(f"Paired tests (only if both one-sample tests significant): variable\n")
        f.write(f"Original α: {bonferroni_info['alpha_original']:.3f}\n")
        f.write(f"Corrected α: {bonferroni_info['alpha_corrected']:.6f}\n\n")
        f.write("="*70 + "\n\n")
        
        for roi, roi_results in results.items():
            if roi == 'bonferroni_info':  # Skip the bonferroni info when iterating through ROIs
                continue
                
            f.write(f"ROI: {roi}\n")
            f.write("-" * 50 + "\n\n")
            
            for network, network_results in roi_results.items():
                f.write(f"Network: {network}\n")
                
                for layer, layer_results in network_results.items():
                    f.write(f"  {layer}:\n")
                    f.write(f"    ONE-SAMPLE T-TESTS (vs 0):\n")
                    f.write(f"      Perceptual: M = {layer_results['perceptual_mean']:.4f}, ")
                    f.write(f"SD = {layer_results['perceptual_std']:.4f}, N = {layer_results['perceptual_n']}\n")
                    f.write(f"        t({layer_results['perceptual_t1samp_df']}) = {layer_results['perceptual_t1samp_stat']:.4f}, ")
                    f.write(f"p = {layer_results['perceptual_t1samp_p']:.4f}")
                    if layer_results['perceptual_t1samp_p'] < bonferroni_info['alpha_corrected']:
                        f.write(" [Significant after Bonferroni correction]")
                    f.write("\n")
                    
                    f.write(f"      Semantic:   M = {layer_results['semantic_mean']:.4f}, ")
                    f.write(f"SD = {layer_results['semantic_std']:.4f}, N = {layer_results['semantic_n']}\n")
                    f.write(f"        t({layer_results['semantic_t1samp_df']}) = {layer_results['semantic_t1samp_stat']:.4f}, ")
                    f.write(f"p = {layer_results['semantic_t1samp_p']:.4f}")
                    if layer_results['semantic_t1samp_p'] < bonferroni_info['alpha_corrected']:
                        f.write(" [Significant after Bonferroni correction]")
                    f.write("\n\n")
                    
                    f.write(f"    PAIRED T-TEST (Perceptual vs Semantic):\n")
                    if layer_results['paired_test_performed']:
                        f.write(f"      t({layer_results['df']}) = {layer_results['t_statistic']:.4f}, ")
                        f.write(f"p = {layer_results['p_value']:.4f}, ")
                        f.write(f"Cohen's d = {layer_results['cohens_d']:.4f}\n")
                        
                        # Add significance interpretation for paired test
                        if layer_results['p_value'] < bonferroni_info['alpha_corrected']:
                            f.write(f"      Significant after Bonferroni correction (p < {bonferroni_info['alpha_corrected']:.6f})\n")
                        else:
                            f.write(f"      Not significant after Bonferroni correction (p ≥ {bonferroni_info['alpha_corrected']:.6f})\n")
                    else:
                        f.write(f"      NOT PERFORMED: Both distributions must be significantly different from zero\n")
                        f.write(f"      (Perceptual p = {layer_results['perceptual_t1samp_p']:.4f}, ")
                        f.write(f"Semantic p = {layer_results['semantic_t1samp_p']:.4f}, ")
                        f.write(f"threshold = {bonferroni_info['alpha_corrected']:.6f})\n")
                    
                    f.write("\n")
                
                f.write("\n")
            f.write("\n")

def main():
    # Load the mediation results
    data_path = '/home/annatruzzi/multiple_deepcluster/results/mediation_proportion_to_total_allsubj_bootstrap.csv'
    output_dir = '/home/annatruzzi/multiple_deepcluster/results/'
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    print("Loading mediation results...")
    df = pd.read_csv(data_path)
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Networks: {df['net'].unique()}")
    print(f"Layers: {df['layer'].unique()}")
    print(f"ROIs: {df['ROI'].unique()}")
    print(f"Types: {df['type'].unique()}")
    
    # Analyze both sum and proportion measures
    measures = ['sum', 'proportion']
    
    for measure in measures:
        print(f"\n\n{'#'*80}")
        print(f"ANALYZING {measure.upper()} VALUES")
        print(f"{'#'*80}")
        
        # Run t-tests
        results = test_perceptual_vs_semantic(df, measure=measure)
        
        # Save results to file
        output_file = os.path.join(output_dir, f'mediation_ttests_{measure}.txt')
        save_results_to_file(results, measure, output_file)
        print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()