import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
import time

class PSOPortfolioOptimizer:
    """
    Portfolio optimization using Particle Swarm Optimization (PSO)
    """
    
    def __init__(self, returns_data, risk_free_rate=0.0, objective='sharpe', bounds=(0, 1)):
        """
        Initialize the PSO optimizer
        
        Parameters:
        -----------
        returns_data : DataFrame
            DataFrame with asset returns (assets in columns, time in index)
        risk_free_rate : float, default=0.0
            Annual risk-free rate for Sharpe ratio calculation
        objective : str, default='sharpe'
            Optimization objective ('sharpe', 'min_risk', 'max_return', 'calmar', or 'sortino') 'sortino')
        bounds : tuple, default=(0, 1)
            Bounds for asset weights (min, max)
        """
        self.returns = returns_data
        self.mean_returns = returns_data.mean()
        self.cov_matrix = returns_data.cov()
        self.num_assets = len(returns_data.columns)
        self.assets = returns_data.columns
        self.risk_free_rate = risk_free_rate / 252  # Convert annual to daily
        self.objective = objective
        self.bounds = bounds
        self.best_weights = None
        self.optimization_result = None
        
    def portfolio_return(self, weights):
        """Calculate expected portfolio return"""
        return np.sum(self.mean_returns * weights)
    
    def portfolio_volatility(self, weights):
        """Calculate portfolio volatility (standard deviation)"""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def portfolio_sharpe_ratio(self, weights):
        """Calculate portfolio Sharpe ratio"""
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        sharpe = (ret - self.risk_free_rate) / vol
        return sharpe
    
    def calculate_max_drawdown(self, weights):
        """Calculate maximum drawdown for the portfolio"""
        portfolio_returns = np.sum(self.returns * weights, axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        return abs(max_drawdown)
    
    def portfolio_calmar_ratio(self, weights):
        """Calculate portfolio Calmar ratio"""
        annual_return = self.portfolio_return(weights) * 252  # Annualize returns
        max_drawdown = self.calculate_max_drawdown(weights)
        # Avoid division by zero
        if max_drawdown == 0:
            return 0
        calmar = annual_return / max_drawdown
        return calmar
    
    def calculate_downside_deviation(self, weights):
        """
        Calculate downside deviation (volatility of negative returns only).
        
        Parameters:
        -----------
        weights : array
            Portfolio weights.
        
        Returns:
        --------
        float
            Downside deviation of the portfolio.
        """
        # Calculate portfolio returns
        portfolio_returns = np.sum(self.returns * weights, axis=1)
        
        # Filter returns below the risk-free rate
        negative_returns = portfolio_returns[portfolio_returns < self.risk_free_rate]
        
        # If no negative returns, downside deviation is zero
        if len(negative_returns) == 0:
            return 0.0
        
        # Calculate downside deviation
        downside_deviation = np.sqrt(np.mean(negative_returns**2))
        return downside_deviation
    
    def portfolio_sortino_ratio(self, weights):
        """Calculate portfolio Sortino ratio"""
        ret = self.portfolio_return(weights)
        downside_dev = self.calculate_downside_deviation(weights)
        # Avoid division by zero
        if downside_dev == 0:
            return 0
        sortino = (ret - self.risk_free_rate) / downside_dev
        return sortino

    def apply_constraints(self, weights):
        """
        Apply constraints to the weights
        - Sum of weights = 1
        - All weights within bounds
        """
        # Apply bounds
        weights = np.clip(weights, self.bounds[0], self.bounds[1])
        
        # Normalize to sum to 1
        weights = weights / np.sum(weights)
        
        return weights
    
    def objective_function(self, x):
        """
        Objective function for PSO optimization
        
        Parameters:
        -----------
        x : array
            Array of shape (n_particles, dimensions)
            
        Returns:
        --------
        array
            Array of objective values for each particle
        """
        n_particles = x.shape[0]
        objective_values = np.zeros(n_particles)
        
        for i in range(n_particles):
            weights = self.apply_constraints(x[i])
            
            if self.objective == 'sharpe':
                # Maximize Sharpe ratio (minimize negative Sharpe)
                objective_values[i] = -self.portfolio_sharpe_ratio(weights)
            elif self.objective == 'min_risk':
                # Minimize volatility
                objective_values[i] = self.portfolio_volatility(weights)
            elif self.objective == 'max_return':
                # Maximize return (minimize negative return)
                objective_values[i] = -self.portfolio_return(weights)
            elif self.objective == 'calmar':
                # Maximize Calmar ratio (minimize negative Calmar)
                objective_values[i] = -self.portfolio_calmar_ratio(weights)
            elif self.objective == 'sortino':
                # Maximize Sortino ratio (minimize negative Sortino)
                objective_values[i] = -self.portfolio_sortino_ratio(weights)
            else:
                raise ValueError(f"Unknown objective: {self.objective}")
                
        return objective_values
    
    def optimize(self, n_particles=100, iters=200, verbose=True):
        """
        Run PSO optimization
        
        Parameters:
        -----------
        n_particles : int, default=100
            Number of particles in the swarm
        iters : int, default=200
            Number of iterations
        verbose : bool, default=True
            Whether to print progress information
            
        Returns:
        --------
        tuple
            (best_weights, expected_return, volatility, sharpe_ratio)
        """
        start_time = time.time()
        
        # PSO options
        options = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}
        
        # Initialize swarm dimensions
        dimensions = self.num_assets
        
        # Define bounds
        min_bound = np.ones(dimensions) * self.bounds[0]
        max_bound = np.ones(dimensions) * self.bounds[1]
        bounds = (min_bound, max_bound)
        
        # Initialize swarm optimizer
        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles, 
            dimensions=dimensions, 
            options=options, 
            bounds=bounds
        )
        
        # Run optimization
        cost, pos = optimizer.optimize(self.objective_function, iters=iters, verbose=verbose)
        
        # Apply constraints to best position
        best_weights = self.apply_constraints(pos)
        
        # Calculate portfolio metrics
        expected_return = self.portfolio_return(best_weights)
        volatility = self.portfolio_volatility(best_weights)
        sharpe_ratio = self.portfolio_sharpe_ratio(best_weights)
        
        # Store results
        self.best_weights = best_weights
        self.optimization_result = {
            'weights': best_weights,
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'cost_history': optimizer.cost_history,
            'execution_time': time.time() - start_time
        }
        
        if verbose:
            print(f"\nPSO Optimization completed in {time.time() - start_time:.2f} seconds")
            print(f"Optimization objective: {self.objective}")
            print(f"Expected Return: {expected_return:.6f}")
            print(f"Volatility: {volatility:.6f}")
            print(f"Sharpe Ratio: {sharpe_ratio:.6f}")
        
        return best_weights, expected_return, volatility, sharpe_ratio
    
    def plot_convergence(self):
        """Plot the convergence of the optimization"""
        if self.optimization_result is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")
            
        cost_history = self.optimization_result['cost_history']
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(cost_history)), cost_history)
        plt.title('PSO Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Value')
        plt.grid(True)
        plt.show()
    
    def plot_weights(self, top_n=10):
        """
        Plot the portfolio weights
        
        Parameters:
        -----------
        top_n : int, default=10
            Number of top holdings to display
        """
        if self.best_weights is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")
            
        # Create DataFrame with weights
        weights_df = pd.DataFrame({
            'Asset': self.assets,
            'Weight': self.best_weights
        })
        
        # Sort by weight descending
        weights_df = weights_df.sort_values('Weight', ascending=False)
        
        # Take top N and aggregate the rest
        if top_n < len(weights_df):
            top_weights = weights_df.iloc[:top_n]
            others_weight = weights_df.iloc[top_n:]['Weight'].sum()
            
            # Create new DataFrame with top N and 'Others'
            # Use pd.concat instead of deprecated append method
            weights_df = pd.concat([
                top_weights,
                pd.DataFrame({'Asset': ['Others'], 'Weight': [others_weight]})
            ])
        
        # Plot pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(weights_df['Weight'], labels=weights_df['Asset'], autopct='%1.1f%%', 
                startangle=90, shadow=True)
        plt.axis('equal')
        plt.title('Portfolio Allocation')
        plt.tight_layout()
        plt.show()
        
    def compare_with_traditional(self, traditional_optimizer):
        """
        Compare PSO results with traditional optimization methods
        
        Parameters:
        -----------
        traditional_optimizer : PortfolioOptimizer
            Instance of traditional portfolio optimizer
            
        Returns:
        --------
        DataFrame
            Comparison of portfolio metrics
        """
        if self.best_weights is None:
            raise ValueError("No PSO optimization has been run yet. Call optimize() first.")
            
        # Get PSO results
        pso_weights = self.best_weights
        pso_return = self.portfolio_return(pso_weights)
        pso_vol = self.portfolio_volatility(pso_weights)
        pso_sharpe = self.portfolio_sharpe_ratio(pso_weights)
        pso_time = self.optimization_result['execution_time']
        
        # Get traditional optimization results
        start_time = time.time()
        if self.objective == 'sharpe':
            trad_weights, trad_return, trad_vol, trad_sharpe = traditional_optimizer.max_sharpe()
        elif self.objective == 'min_risk':
            trad_weights, trad_return, trad_vol, trad_sharpe = traditional_optimizer.min_variance()
        else:
            # For max_return, use maximum return from efficient frontier
            ef_returns, ef_vols, ef_weights = traditional_optimizer.efficient_frontier(10)
            max_return_idx = np.argmax(ef_returns)
            trad_weights = ef_weights[max_return_idx]
            trad_return = ef_returns[max_return_idx]
            trad_vol = ef_vols[max_return_idx]
            trad_sharpe = (trad_return - self.risk_free_rate) / trad_vol
        
        trad_time = time.time() - start_time
        
        # Build comparison DataFrame
        comparison = pd.DataFrame({
            'PSO': [pso_return, pso_vol, pso_sharpe, pso_time],
            'Traditional': [trad_return, trad_vol, trad_sharpe, trad_time]
        }, index=['Expected Return', 'Volatility', 'Sharpe Ratio', 'Execution Time (s)'])
        
        # Calculate difference as percentage
        comparison['Difference (%)'] = (comparison['PSO'] / comparison['Traditional'] - 1) * 100
        
        return comparison
    
    def backtest_portfolio(self, initial_investment=1000, test_data=None, plot=True):
        """
        Backtest portfolio performance using the optimized weights
        
        Parameters:
        -----------
        initial_investment : float, default=1000
            Initial investment amount
        test_data : DataFrame, optional
            Test data for out-of-sample testing. If None, uses full dataset
        plot : bool, default=True
            Whether to plot the equity curves
            
        Returns:
        --------
        dict
            Dictionary containing backtest results
        """
        if self.best_weights is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")
            
        # Store results for both datasets
        results = {}
        
        # Backtest on full dataset (in-sample)
        full_returns = np.sum(self.returns * self.best_weights, axis=1)
        full_cumulative = (1 + full_returns).cumprod()
        results['full_equity'] = initial_investment * full_cumulative
        
        # Backtest on test data if provided (out-of-sample)
        if test_data is not None:
            test_returns = np.sum(test_data * self.best_weights, axis=1)
            test_cumulative = (1 + test_returns).cumprod()
            results['test_equity'] = initial_investment * test_cumulative
        
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(results['full_equity'].index, results['full_equity'], 
                    label=f'Full Dataset (Final: ${results["full_equity"].iloc[-1]:.2f})')
            
            if test_data is not None:
                plt.plot(results['test_equity'].index, results['test_equity'], 
                        label=f'Test Set (Final: ${results["test_equity"].iloc[-1]:.2f})')
            
            plt.title(f'Portfolio Equity Curve (Initial: ${initial_investment:,.2f})')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        return results

# Example usage
if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    from src.data.data_processor import DataProcessor
    from src.optimization.portfolio_theory import PortfolioOptimizer
    
    # Initialize data processor
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "5m")
    data_processor = DataProcessor(data_dir)
    
    # Load stock data (limit to 20 stocks for example)
    data_processor.load_stock_data(limit=20)
    
    # Calculate daily returns
    daily_returns = data_processor.calculate_daily_returns()
    
    # Initialize PSO optimizer
    pso_optimizer = PSOPortfolioOptimizer(daily_returns, risk_free_rate=0.02, objective='sharpe')
    
    # Run PSO optimization
    pso_optimizer.optimize(n_particles=50, iters=100)
    
    # Plot convergence
    pso_optimizer.plot_convergence()
    
    # Plot weights
    pso_optimizer.plot_weights()
    
    # Compare with traditional optimization
    traditional_optimizer = PortfolioOptimizer(daily_returns, risk_free_rate=0.02)
    comparison = pso_optimizer.compare_with_traditional(traditional_optimizer)
    
    print("\nComparison with Traditional Optimization:")
    print(comparison)
    
    # Backtest portfolio
    backtest_results = pso_optimizer.backtest_portfolio(initial_investment=1000, plot=True)
    print("\nBacktest Results:")
    print(backtest_results)