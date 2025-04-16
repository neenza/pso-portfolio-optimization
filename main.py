import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.data.data_processor import DataProcessor
from src.optimization.portfolio_theory import PortfolioOptimizer
from src.optimization.pso_optimizer import PSOPortfolioOptimizer

def main():
    """Main function to run portfolio optimization"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Portfolio Optimization using PSO')
    parser.add_argument('--data_dir', type=str, default='5m',
                       help='Directory containing stock data CSV files')
    parser.add_argument('--n_stocks', type=int, default=30,
                       help='Number of stocks to include in portfolio (default: 30)')
    parser.add_argument('--risk_free_rate', type=float, default=0.02,
                       help='Annual risk-free rate (default: 0.02 or 2%)')
    parser.add_argument('--objective', type=str, default='sharpe', 
                       choices=['sharpe', 'min_risk', 'max_return', 'calmar', 'sortino'],
                       help='Optimization objective (default: sharpe)')
    parser.add_argument('--n_particles', type=int, default=100,
                       help='Number of particles in PSO (default: 100)')
    parser.add_argument('--iters', type=int, default=200,
                       help='Number of iterations in PSO (default: 200)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results (default: results)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Print settings
    print("Portfolio Optimization Settings:")
    print(f"Data directory: {args.data_dir}")
    print(f"Number of stocks: {args.n_stocks}")
    print(f"Risk-free rate: {args.risk_free_rate * 100}%")
    print(f"Optimization objective: {args.objective}")
    print(f"PSO particles: {args.n_particles}")
    print(f"PSO iterations: {args.iters}")
    print(f"Output directory: {args.output_dir}")
    print("\n" + "="*50 + "\n")
    
    # Step 1: Load and process data
    print("Step 1: Loading and processing stock data")
    data_processor = DataProcessor(args.data_dir)
    stock_data = data_processor.load_stock_data(limit=args.n_stocks)
    daily_returns = data_processor.calculate_daily_returns()
    
    # Calculate correlation matrix
    correlation_matrix = data_processor.calculate_correlation_matrix()
    
    # Split data into training and testing sets
    train_data, test_data = data_processor.split_train_test(test_ratio=0.2)
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    
    # Step 2: Create traditional portfolio optimizer
    print("\nStep 2: Running traditional portfolio optimization")
    traditional_optimizer = PortfolioOptimizer(train_data, risk_free_rate=args.risk_free_rate)
    
    # Calculate metrics for different portfolio strategies
    weights_df, metrics_df = traditional_optimizer.portfolio_metrics_report()
    
    print("\nPortfolio Metrics (Traditional Optimization):")
    print(metrics_df)
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(args.output_dir, 'traditional_metrics.csv'))
    weights_df.to_csv(os.path.join(args.output_dir, 'traditional_weights.csv'))
    
    # Step 3: Run PSO optimization
    print("\nStep 3: Running PSO portfolio optimization")
    pso_optimizer = PSOPortfolioOptimizer(
        train_data, 
        risk_free_rate=args.risk_free_rate, 
        objective=args.objective
    )
    
    pso_weights, pso_return, pso_vol, pso_sharpe = pso_optimizer.optimize(
        n_particles=args.n_particles, 
        iters=args.iters
    )
    
    # Save PSO weights to CSV
    pd.DataFrame({
        'Asset': train_data.columns,
        'Weight': pso_weights
    }).to_csv(os.path.join(args.output_dir, 'pso_weights.csv'), index=False)
    
    # Step 4: Compare results
    print("\nStep 4: Comparing optimization methods")
    comparison = pso_optimizer.compare_with_traditional(traditional_optimizer)
    print("\nComparison with Traditional Optimization:")
    print(comparison)
    
    # Save comparison to CSV
    comparison.to_csv(os.path.join(args.output_dir, 'optimization_comparison.csv'))
    
    # Step 5: Visualize results
    print("\nStep 5: Generating visualizations")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    pso_optimizer.plot_convergence()
    plt.savefig(os.path.join(args.output_dir, 'convergence_plot.png'))
    
    # Plot portfolio weights
    plt.figure(figsize=(12, 8))
    pso_optimizer.plot_weights(top_n=10)
    plt.savefig(os.path.join(args.output_dir, 'portfolio_allocation.png'))
    
    # Plot efficient frontier
    plt.figure(figsize=(12, 8))
    traditional_optimizer.plot_efficient_frontier(show_assets=True, show_min_vol=True, show_max_sharpe=True)
    
    # Add PSO portfolio to the efficient frontier plot
    plt.scatter(pso_vol, pso_return, marker='*', color='purple', s=200, label='PSO Portfolio')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'efficient_frontier.png'))
    
    # Plot correlation heatmap
    plt.figure(figsize=(14, 12))
    data_processor.plot_correlation_heatmap(n_stocks=min(20, args.n_stocks))
    plt.savefig(os.path.join(args.output_dir, 'correlation_heatmap.png'))
    
    print(f"\nAll results saved to {args.output_dir} directory")
    
    # Step 6: Backtest portfolios
    print("\nStep 6: Backtesting portfolios")
    
    # Get initial equal weights (before optimization)
    initial_weights = np.ones(len(daily_returns.columns)) / len(daily_returns.columns)
    
    # Initialize with $1000 investment
    initial_investment = 1000
    
    # Calculate returns using initial weights on full dataset
    full_returns_initial = np.sum(daily_returns * initial_weights, axis=1)
    full_cumulative_initial = (1 + full_returns_initial).cumprod()
    initial_equity = initial_investment * full_cumulative_initial
    
    # Calculate returns using PSO weights on full dataset
    full_returns_optimized = np.sum(daily_returns * pso_weights, axis=1)
    full_cumulative_optimized = (1 + full_returns_optimized).cumprod()
    optimized_equity = initial_investment * full_cumulative_optimized

    # Create two subplots - one for absolute values and one for percentage change
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Plot absolute equity curves
    ax1.plot(initial_equity.index, initial_equity, 
            label=f'Initial Equal Weights (Final: ${initial_equity.iloc[-1]:.2f})')
    
    ax1.plot(optimized_equity.index, optimized_equity,
            label=f'Optimized Weights (Final: ${optimized_equity.iloc[-1]:.2f})')
    
    initial_final = initial_equity.iloc[-1]
    optimized_final = optimized_equity.iloc[-1]
    
    # Add annotations for final values
    ax1.annotate(f'${initial_final:.2f}', 
                xy=(initial_equity.index[-1], initial_final),
                xytext=(5, 0), textcoords='offset points',
                fontsize=10)
                
    ax1.annotate(f'${optimized_final:.2f}', 
                xy=(optimized_equity.index[-1], optimized_final),
                xytext=(5, 0), textcoords='offset points',
                fontsize=10)
    
    ax1.set_title(f'Portfolio Equity Curves Comparison (Initial: ${initial_investment:,.2f})')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot percentage change from initial investment
    initial_pct = (initial_equity / initial_investment - 1) * 100
    optimized_pct = (optimized_equity / initial_investment - 1) * 100
    
    ax2.plot(initial_pct.index, initial_pct, 
            label=f'Initial Equal Weights (+{initial_pct.iloc[-1]:.2f}%)')
    
    ax2.plot(optimized_pct.index, optimized_pct,
            label=f'Optimized Weights (+{optimized_pct.iloc[-1]:.2f}%)')
    
    # Add annotations for final percentage changes
    ax2.annotate(f'+{initial_pct.iloc[-1]:.2f}%', 
                xy=(initial_pct.index[-1], initial_pct.iloc[-1]),
                xytext=(5, 0), textcoords='offset points',
                fontsize=10)
                
    ax2.annotate(f'+{optimized_pct.iloc[-1]:.2f}%', 
                xy=(optimized_pct.index[-1], optimized_pct.iloc[-1]),
                xytext=(5, 0), textcoords='offset points',
                fontsize=10)
    
    ax2.set_title('Portfolio Percentage Change from Initial Investment')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Percentage Change (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'equity_curves_comparison.png'))
    plt.close()

    # Calculate and save performance metrics
    metrics = ['Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']
    perf_comparison = pd.DataFrame(index=metrics)
    
    # Calculate metrics for both portfolios using full dataset
    for name, equity in [('Initial Weights', initial_equity), 
                        ('Optimized Weights', optimized_equity)]:
        returns = equity.pct_change().dropna()
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        ann_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = (ann_return - args.risk_free_rate) / volatility
        drawdown = (equity / equity.expanding().max() - 1).min()
        
        perf_comparison[name] = [total_return, ann_return, volatility, sharpe, drawdown]
    
    print("\nFull Dataset Performance Comparison:")
    print(perf_comparison)
    
    # Calculate and print the ratio of optimized return to initial return
    return_ratio = perf_comparison.loc['Total Return', 'Optimized Weights'] / perf_comparison.loc['Total Return', 'Initial Weights']
    print(f"\nOptimized portfolio return is {return_ratio:.2f}x the initial equal-weighted portfolio return")
    
    # Save performance comparison
    perf_comparison.to_csv(os.path.join(args.output_dir, 'full_dataset_performance.csv'))
    
    print("\nPortfolio optimization complete!")

if __name__ == "__main__":
    main()